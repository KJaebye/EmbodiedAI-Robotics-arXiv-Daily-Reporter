# A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation 

**Title (ZH)**: 基于大语言模型的综合驱动智能化评估框架 

**Authors**: Shanhe You, Xuewen Luo, Xinhe Liang, Jiashu Yu, Chen Zheng, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05164)  

**Abstract**: Evaluation methods for autonomous driving are crucial for algorithm optimization. However, due to the complexity of driving intelligence, there is currently no comprehensive evaluation method for the level of autonomous driving intelligence. In this paper, we propose an evaluation framework for driving behavior intelligence in complex traffic environments, aiming to fill this gap. We constructed a natural language evaluation dataset of human professional drivers and passengers through naturalistic driving experiments and post-driving behavior evaluation interviews. Based on this dataset, we developed an LLM-powered driving evaluation framework. The effectiveness of this framework was validated through simulated experiments in the CARLA urban traffic simulator and further corroborated by human assessment. Our research provides valuable insights for evaluating and designing more intelligent, human-like autonomous driving agents. The implementation details of the framework and detailed information about the dataset can be found at Github. 

**Abstract (ZH)**: 自动驾驶驾驶行为智能评价方法对于算法优化至关重要。然而，由于驾驶智能化的复杂性，目前尚无全面的自动驾驶智能化水平评价方法。本文提出了一种用于复杂交通环境下的驾驶行为智能评价框架，旨在填补这一空白。我们通过自然驾驶实验和驾驶后行为评估访谈，构建了一个基于自然语言的人类专业司机和乘客的评价数据集。基于该数据集，我们开发了一种基于大语言模型的驾驶评价框架。通过CARLA城市交通模拟器的模拟实验和进一步的人工评估验证了该框架的有效性。我们的研究为评估和设计更具智能化、更像人类的自动驾驶代理提供了有价值的见解。框架的实现细节和数据集的详细信息可在GitHub上找到。 

---
# R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning 

**Title (ZH)**: R1-Searcher: 通过强化学习激励大语言模型的搜索能力 

**Authors**: Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.05592)  

**Abstract**: Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini. 

**Abstract (ZH)**: 现有的大型推理模型(LRMs)展示了强化学习(RL)在提升大型语言模型(LLMs)的复杂推理能力方面的潜力。尽管它们在数学和编程等挑战性任务上取得了显著性能，但在处理时间敏感或知识密集的问题时，常常依赖内部知识，这可能导致不准确性和幻觉。为了解决这一问题，我们提出了一种名为R1-Searcher的新型两阶段基于结果的RL方法，旨在增强LLMs的搜索能力。该方法使LLMs能够在推理过程中自主调用外部搜索系统以访问额外知识。我们的框架仅依赖于RL，无需过程奖励或冷启动的蒸馏。实验结果表明，我们的方法在性能上显著优于之前的强大检索增強方法，甚至优于闭源的GPT-4o-mini。 

---
# Ontology Generation using Large Language Models 

**Title (ZH)**: 基于大型语言模型的本体生成 

**Authors**: Anna Sofia Lippolis, Mohammad Javad Saeedizade, Robin Keskisärkkä, Sara Zuppiroli, Miguel Ceriani, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2503.05388)  

**Abstract**: The ontology engineering process is complex, time-consuming, and error-prone, even for experienced ontology engineers. In this work, we investigate the potential of Large Language Models (LLMs) to provide effective OWL ontology drafts directly from ontological requirements described using user stories and competency questions. Our main contribution is the presentation and evaluation of two new prompting techniques for automated ontology development: Memoryless CQbyCQ and Ontogenia. We also emphasize the importance of three structural criteria for ontology assessment, alongside expert qualitative evaluation, highlighting the need for a multi-dimensional evaluation in order to capture the quality and usability of the generated ontologies. Our experiments, conducted on a benchmark dataset of ten ontologies with 100 distinct CQs and 29 different user stories, compare the performance of three LLMs using the two prompting techniques. The results demonstrate improvements over the current state-of-the-art in LLM-supported ontology engineering. More specifically, the model OpenAI o1-preview with Ontogenia produces ontologies of sufficient quality to meet the requirements of ontology engineers, significantly outperforming novice ontology engineers in modelling ability. However, we still note some common mistakes and variability of result quality, which is important to take into account when using LLMs for ontology authoring support. We discuss these limitations and propose directions for future research. 

**Abstract (ZH)**: 大型语言模型在直接从用户故事和专业问题生成OWL本体草案中的潜力研究：Memoryless CQbyCQ和Ontogenia方法及其评估 

---
# WritingBench: A Comprehensive Benchmark for Generative Writing 

**Title (ZH)**: WritingBench: 生成写作的综合基准 

**Authors**: Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, SHaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05244)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced text generation capabilities, yet evaluating their performance in generative writing remains a challenge. Existing benchmarks primarily focus on generic text generation or limited in writing tasks, failing to capture the diverse requirements of high-quality written contents across various domains. To bridge this gap, we present WritingBench, a comprehensive benchmark designed to evaluate LLMs across 6 core writing domains and 100 subdomains, encompassing creative, persuasive, informative, and technical writing. We further propose a query-dependent evaluation framework that empowers LLMs to dynamically generate instance-specific assessment criteria. This framework is complemented by a fine-tuned critic model for criteria-aware scoring, enabling evaluations in style, format and length. The framework's validity is further demonstrated by its data curation capability, which enables 7B-parameter models to approach state-of-the-art (SOTA) performance. We open-source the benchmark, along with evaluation tools and modular framework components, to advance the development of LLMs in writing. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）显著提升了文本生成能力，但评估其在生成性写作中的表现依然颇具挑战。现有基准主要集中在通用文本生成或局限于某些写作任务，未能捕捉各领域高质量书面内容的多样化需求。为弥补这一不足，我们提出了WritingBench，这是一个综合基准，旨在评估LLMs在6个核心写作领域和100个子领域中的表现，涵盖创造性、 persuasiveness、信息性和技术性写作。我们还提出了一个查询依赖的评估框架，使LLMs能够动态生成实例特定的评估标准。该框架结合了一个细调的评论者模型，用于根据标准评分，从而实现风格、格式和长度的评估。框架的有效性通过其数据整理能力得以证明，使7B参数模型接近了当前最佳表现（SOTA）。我们开源了该基准以及评估工具和模块化框架组件，以促进大型语言模型在写作方面的开发。 

---
# Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning 

**Title (ZH)**: 符号混合专家：异构推理的自适应技能路由 

**Authors**: Justin Chih-Yao Chen, Sukwon Yun, Elias Stengel-Eskin, Tianlong Chen, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.05641)  

**Abstract**: Combining existing pre-trained expert LLMs is a promising avenue for scalably tackling large-scale and diverse tasks. However, selecting experts at the task level is often too coarse-grained, as heterogeneous tasks may require different expertise for each instance. To enable adaptive instance-level mixing of pre-trained LLM experts, we propose Symbolic-MoE, a symbolic, text-based, and gradient-free Mixture-of-Experts framework. Symbolic-MoE takes a fine-grained approach to selection by emphasizing skills, e.g., algebra in math or molecular biology in biomedical reasoning. We propose a skill-based recruiting strategy that dynamically selects the most relevant set of expert LLMs for diverse reasoning tasks based on their strengths. Each selected expert then generates its own reasoning, resulting in k outputs from k experts, which are then synthesized into a final high-quality response by an aggregator chosen based on its ability to integrate diverse reasoning outputs. We show that Symbolic-MoE's instance-level expert selection improves performance by a large margin but -- when implemented naively -- can introduce a high computational overhead due to the need for constant model loading and offloading. To address this, we implement a batch inference strategy that groups instances based on their assigned experts, loading each model only once. This allows us to integrate 16 expert models on 1 GPU with a time cost comparable to or better than prior multi-agent baselines using 4 GPUs. Through extensive evaluations on diverse benchmarks (MMLU-Pro, GPQA, AIME, and MedMCQA), we demonstrate that Symbolic-MoE outperforms strong LLMs like GPT4o-mini, as well as multi-agent approaches, with an absolute average improvement of 8.15% over the best multi-agent baseline. Moreover, Symbolic-MoE removes the need for expensive multi-round discussions, outperforming discussion baselines with less computation. 

**Abstract (ZH)**: 结合现有的预训练专家大规模语言模型是一种有望应对大规模和多样化任务的途径。然而，任务级别的专家选择往往过于粗粒度，因为异构任务可能需要每个实例有不同的专长。为了实现预训练语言模型专家的自适应实例级混合，我们提出了一种符号混合-of-专家（Symbolic-MoE）框架，该框架是符号的、基于文本的和无需梯度的混合-of-专家框架。Symbolic-MoE 通过强调技能，如数学中的代数或生物医学推理中的分子生物学，采取了细粒度的选择方法。我们提出了一种基于技能的招聘策略，该策略根据专家的长处动态选择最适合多种推理任务的专家集。每个选定的专家生成其自的推理，从而产生来自k个专家的k个输出，这些输出然后由根据其整合多样推理输出的能力选择的聚合器进行综合生成最终的高质量响应。我们展示了Symbolic-MoE的实例级专家选择能够显著提升性能，但未经优化时可能会因需要不断加载和卸载模型而引入高计算开销。为了解决这个问题，我们实施了批量推理策略，根据分配的专家对实例进行分组，并且只加载每个模型一次。这使得我们能够在1个GPU上整合16个专家模型，计算成本与4个GPU的多智能体基线相当或更优。通过在多样基准（MMLU-Pro、GPQA、AIME和MedMCQA）上的 extensive 评估，我们证明了Symbolic-MoE 在性能上优于强大的语言模型如GPT4o-mini，以及多智能体方法，绝对平均改进率为8.15%，并且优于最好的多智能体基线。此外，Symbolic-MoE 消除了昂贵的多轮讨论需求，并在计算开销较少的情况下优于讨论基线。 

---
# Learning LLM Preference over Intra-Dialogue Pairs: A Framework for Utterance-level Understandings 

**Title (ZH)**: 学习大规模语言模型在对话内部配对中的偏好：一种语句级理解的框架 

**Authors**: Xuanqing Liu, Luyang Kong, Wei Niu, Afshin Khashei, Belinda Zeng, Steve Johnson, Jon Jay, Davor Golac, Matt Pope  

**Link**: [PDF](https://arxiv.org/pdf/2503.05620)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in handling complex dialogue tasks without requiring use case-specific fine-tuning. However, analyzing live dialogues in real-time necessitates low-latency processing systems, making it impractical to deploy models with billions of parameters due to latency constraints. As a result, practitioners often prefer smaller models with millions of parameters, trained on high-quality, human-annotated datasets. Yet, curating such datasets is both time-consuming and costly. Consequently, there is a growing need to combine the scalability of LLM-generated labels with the precision of human annotations, enabling fine-tuned smaller models to achieve both higher speed and accuracy comparable to larger models. In this paper, we introduce a simple yet effective framework to address this challenge. Our approach is specifically designed for per-utterance classification problems, which encompass tasks such as intent detection, dialogue state tracking, and more. To mitigate the impact of labeling errors from LLMs -- the primary source of inaccuracies in student models -- we propose a noise-reduced preference learning loss. Experimental results demonstrate that our method significantly improves accuracy across utterance-level dialogue tasks, including sentiment detection (over $2\%$), dialogue act classification (over $1.5\%$), etc. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在无需特定用例微调的情况下展示了处理复杂对话任务的非凡能力。然而，实时分析对话需要低延迟处理系统，因此由于延迟限制，部署具有 billions 参数的模型变得不切实际。因此，实践者通常更倾向于使用 millions 参数的小型模型，并在高质量的人工标注数据集上进行训练。然而，收集这样的数据集既耗时又昂贵。因此，迫切需要结合LLM生成标签的可扩展性与人工标注的精确性，从而使微调后的较小模型能够同时在速度和准确率上与较大模型匹敌。在本文中，我们介绍了一种简单且有效的框架来应对这一挑战。我们的方法特别适用于每句话分类问题，包括意图检测、对话状态跟踪等任务。为了减轻来自LLM的标注错误对学生模型准确性的影响——主要来源——我们提出了一种噪声减少的偏好学习损失。实验结果表明，我们的方法在包括情感识别（超过2%）、对话行为分类（超过1.5%）等句子级对话任务上的准确性显著提升。 

---
# A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models 

**Title (ZH)**: Sparse 自编码器研究：解读大型语言模型的内部机制 

**Authors**: Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai, Ziyu Yao, Ninghao Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.05613)  

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, yet their internal mechanisms remain largely opaque. Recently, mechanistic interpretability has attracted significant attention from the research community as a means to understand the inner workings of LLMs. Among various mechanistic interpretability approaches, Sparse Autoencoders (SAEs) have emerged as a particularly promising method due to their ability to disentangle the complex, superimposed features within LLMs into more interpretable components. This paper presents a comprehensive examination of SAEs as a promising approach to interpreting and understanding LLMs. We provide a systematic overview of SAE principles, architectures, and applications specifically tailored for LLM analysis, covering theoretical foundations, implementation strategies, and recent developments in sparsity mechanisms. We also explore how SAEs can be leveraged to explain the internal workings of LLMs, steer model behaviors in desired directions, and develop more transparent training methodologies for future models. Despite the challenges that remain around SAE implementation and scaling, they continue to provide valuable tools for understanding the internal mechanisms of large language models. 

**Abstract (ZH)**: 大型语言模型的机制解释：稀疏自编码器在理解大型语言模型中的应用 

---
# AceWGS: An LLM-Aided Framework to Accelerate Catalyst Design for Water-Gas Shift Reactions 

**Title (ZH)**: AceWGS：一种基于LLM的框架，用于加速水煤气转变反应催化剂设计 

**Authors**: Joyjit Chattoraj, Brahim Hamadicharef, Teo Shi Chang, Yingzhi Zeng, Chee Kok Poh, Luwei Chen, Teck Leong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.05607)  

**Abstract**: While the Water-Gas Shift (WGS) reaction plays a crucial role in hydrogen production for fuel cells, finding suitable catalysts to achieve high yields for low-temperature WGS reactions remains a persistent challenge. Artificial Intelligence (AI) has shown promise in accelerating catalyst design by exploring vast candidate spaces, however, two key gaps limit its effectiveness. First, AI models primarily train on numerical data, which fail to capture essential text-based information, such as catalyst synthesis methods. Second, the cross-disciplinary nature of catalyst design requires seamless collaboration between AI, theory, experiments, and numerical simulations, often leading to communication barriers. To address these gaps, we present AceWGS, a Large Language Models (LLMs)-aided framework to streamline WGS catalyst design. AceWGS interacts with researchers through natural language, answering queries based on four features: (i) answering general queries, (ii) extracting information about the database comprising WGS-related journal articles, (iii) comprehending the context described in these articles, and (iv) identifying catalyst candidates using our proposed AI inverse model. We presented a practical case study demonstrating how AceWGS can accelerate the catalyst design process. AceWGS, built with open-source tools, offers an adjustable framework that researchers can readily adapt for a range of AI-accelerated catalyst design applications, supporting seamless integration across cross-disciplinary studies. 

**Abstract (ZH)**: AceWGS：一种大型语言模型辅助的水煤气变换催化剂设计框架 

---
# Quantifying the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data 

**Title (ZH)**: 量化检索增强语言模型在地面数据中虚假特征面前的 robustness 

**Authors**: Shiping Yang, Jie Wu, Wenbiao Ding, Ning Wu, Shining Liang, Ming Gong, Hengyuan Zhang, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05587)  

**Abstract**: Robustness has become a critical attribute for the deployment of RAG systems in real-world applications. Existing research focuses on robustness to explicit noise (e.g., document semantics) but overlooks spurious features (a.k.a. implicit noise). While previous works have explored spurious features in LLMs, they are limited to specific features (e.g., formats) and narrow scenarios (e.g., ICL). In this work, we statistically confirm the presence of spurious features in the RAG paradigm, a robustness problem caused by the sensitivity of LLMs to semantic-agnostic features. Moreover, we provide a comprehensive taxonomy of spurious features and empirically quantify their impact through controlled experiments. Further analysis reveals that not all spurious features are harmful and they can even be beneficial sometimes. Extensive evaluation results across multiple LLMs suggest that spurious features are a widespread and challenging problem in the field of RAG. The code and dataset will be released to facilitate future research. We release all codes and data at: $\\\href{this https URL}{this https URL}$. 

**Abstract (ZH)**: 鲁棒性已成为在实际应用中部署RAG系统的关键属性。现有研究主要关注显式噪声（例如，文档语义）的鲁棒性，但忽视了隐性特征（即隐式噪声）。虽然之前的工作已经在LLMs中探索了隐性特征，但这些工作局限于特定的特征（例如，格式）和狭窄的场景（例如，ICL）。在本文中，我们通过统计方法证实了在RAG范式中存在隐性特征，这是一种由LLMs对语义无关特征的敏感性引起的鲁棒性问题。此外，我们提供了隐性特征的全面分类，并通过受控实验实证衡量其影响。进一步的分析表明，并非所有隐性特征都是有害的，有时它们甚至可能是有益的。在多个LLMs上的广泛评估结果表明，隐性特征是RAG领域中普遍存在且具有挑战性的问题。我们将发布代码和数据以促进未来的研究：$\\\href{this https URL}{this https URL}$。 

---
# Cognitive Bias Detection Using Advanced Prompt Engineering 

**Title (ZH)**: 高级提示工程在认知偏差检测中的应用 

**Authors**: Frederic Lemieux, Aisha Behr, Clara Kellermann-Bryant, Zaki Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2503.05516)  

**Abstract**: Cognitive biases, systematic deviations from rationality in judgment, pose significant challenges in generating objective content. This paper introduces a novel approach for real-time cognitive bias detection in user-generated text using large language models (LLMs) and advanced prompt engineering techniques. The proposed system analyzes textual data to identify common cognitive biases such as confirmation bias, circular reasoning, and hidden assumption. By designing tailored prompts, the system effectively leverages LLMs' capabilities to both recognize and mitigate these biases, improving the quality of human-generated content (e.g., news, media, reports). Experimental results demonstrate the high accuracy of our approach in identifying cognitive biases, offering a valuable tool for enhancing content objectivity and reducing the risks of biased decision-making. 

**Abstract (ZH)**: 基于大型语言模型和高级提示工程的实时用户生成文本认知偏差检测方法 

---
# Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs? 

**Title (ZH)**: 基于语法的代码表示：LLM值得追求的目标吗？ 

**Authors**: Qingyuan Liang, Zhao Zhang, Zeyu Sun, Zheng Lin, Qi Luo, Yueyi Xiao, Yizhou Chen, Yuqun Zhang, Haotian Zhang, Lu Zhang, Bin Chen, Yingfei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.05507)  

**Abstract**: Grammar serves as a cornerstone in programming languages and software engineering, providing frameworks to define the syntactic space and program structure. Existing research demonstrates the effectiveness of grammar-based code representations in small-scale models, showing their ability to reduce syntax errors and enhance performance. However, as language models scale to the billion level or beyond, syntax-level errors become rare, making it unclear whether grammar information still provides performance benefits. To explore this, we develop a series of billion-scale GrammarCoder models, incorporating grammar rules in the code generation process. Experiments on HumanEval (+) and MBPP (+) demonstrate a notable improvement in code generation accuracy. Further analysis shows that grammar-based representations enhance LLMs' ability to discern subtle code differences, reducing semantic errors caused by minor variations. These findings suggest that grammar-based code representations remain valuable even in billion-scale models, not only by maintaining syntax correctness but also by improving semantic differentiation. 

**Abstract (ZH)**: 语法在编程语言和软件工程中作为 foundation，提供定义语法空间和程序结构的框架。现有研究显示，基于语法的代码表示在小型模型中具有有效性，能够减少语法错误并提升性能。然而，当语言模型扩展到亿级或更大规模时，语法级别错误变得罕见，这使得不清楚语法信息是否仍能提供性能上的优势。为了探索这一点，我们开发了一系列亿级规模的GrammarCoder模型，在代码生成过程中融入了语法规则。在HumanEval (+) 和 MBPP (+) 上的实验显示代码生成精度有显著提升。进一步的分析表明，基于语法的表示增强了大型语言模型区分细微代码差异的能力，减少了由细微差异引起的语义错误。这些发现表明，即使在亿级模型中，基于语法的代码表示仍然具有价值，不仅通过保持语法的正确性，还通过提高语义区分能力来提升性能。 

---
# Soft Policy Optimization: Online Off-Policy RL for Sequence Models 

**Title (ZH)**: 软策略优化：在线离策序贯模型强化学习 

**Authors**: Taco Cohen, David W. Zhang, Kunhao Zheng, Yunhao Tang, Remi Munos, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2503.05453)  

**Abstract**: RL-based post-training of language models is almost exclusively done using on-policy methods such as PPO. These methods cannot learn from arbitrary sequences such as those produced earlier in training, in earlier runs, by human experts or other policies, or by decoding and exploration methods. This results in severe sample inefficiency and exploration difficulties, as well as a potential loss of diversity in the policy responses. Moreover, asynchronous PPO implementations require frequent and costly model transfers, and typically use value models which require a large amount of memory. In this paper we introduce Soft Policy Optimization (SPO), a simple, scalable and principled Soft RL method for sequence model policies that can learn from arbitrary online and offline trajectories and does not require a separate value model. In experiments on code contests, we shows that SPO outperforms PPO on pass@10, is significantly faster and more memory efficient, is able to benefit from off-policy data, enjoys improved stability, and learns more diverse (i.e. soft) policies. 

**Abstract (ZH)**: 基于RL的后训练语言模型几乎完全使用了基于策略的方法（如PPO）。这些方法无法从训练早期、先前运行、人类专家或其他策略、或解码和探索方法生成的任意序列中学习，这导致了严重的样本效率低下和探索困难，并可能损失策略响应的多样性。此外，异步PPO实现需要频繁且昂贵的模型转移，并通常使用需要大量内存的价值模型。在本文中，我们引入了Soft Policy Optimization (SPO)，这是一种简单、可扩展且基于原则的软RL方法，用于序列模型策略，可以从任意的在线和离线轨迹中学习，且不需要独立的价值模型。在代码竞赛实验中，我们展示了SPO在pass@10上优于PPO，速度快且内存效率更高，能够利用离策略数据，具有更好的稳定性，并学习到更具多样性的（即软的）政策。 

---
# LLM-based Iterative Approach to Metamodeling in Automotive 

**Title (ZH)**: 基于LLM的迭代元建模方法在汽车领域中的应用 

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.05449)  

**Abstract**: In this paper, we introduce an automated approach to domain-specific metamodel construction relying on Large Language Model (LLM). The main focus is adoption in automotive domain. As outcome, a prototype was implemented as web service using Python programming language, while OpenAI's GPT-4o was used as the underlying LLM. Based on the initial experiments, this approach successfully constructs Ecore metamodel based on set of automotive requirements and visualizes it making use of PlantUML notation, so human experts can provide feedback in order to refine the result. Finally, locally deployable solution is also considered, including the limitations and additional steps required. 

**Abstract (ZH)**: 基于大型语言模型的汽车领域特定元模型自动化构建方法 

---
# An Empirical Study of Conformal Prediction in LLM with ASP Scaffolds for Robust Reasoning 

**Title (ZH)**: 基于ASP支架的LLM中一致预测的实证研究：稳健推理 

**Authors**: Navdeep Kaur, Lachlan McPheat, Alessandra Russo, Anthony G Cohn, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2503.05439)  

**Abstract**: In this paper, we examine the use of Conformal Language Modelling (CLM) alongside Answer Set Programming (ASP) to enhance the performance of standard open-weight LLMs on complex multi-step reasoning tasks. Using the StepGame dataset, which requires spatial reasoning, we apply CLM to generate sets of ASP programs from an LLM, providing statistical guarantees on the correctness of the outputs. Experimental results show that CLM significantly outperforms baseline models that use standard sampling methods, achieving substantial accuracy improvements across different levels of reasoning complexity. Additionally, the LLM-as-Judge metric enhances CLM's performance, especially in assessing structurally and logically correct ASP outputs. However, calibrating CLM with diverse calibration sets did not improve generalizability for tasks requiring much longer reasoning steps, indicating limitations in handling more complex tasks. 

**Abstract (ZH)**: 本文探究将同构语言模型(CLM)与回答集编程(ASP)结合使用以增强标准开放权重语言模型在复杂多步推理任务中的性能。通过StepGame数据集，该数据集需要空间推理能力，我们将CLM应用于从语言模型生成ASP程序集，并提供输出正确性的统计保证。实验结果表明，CLM显著优于使用标准采样方法的基线模型，在不同的推理复杂度级别上实现了显著的准确率提升。此外，LLM作为裁判的评估指标进一步提升了CLM的性能，特别是在评估结构上和逻辑上正确的ASP输出方面尤为明显。然而，使用多样化的校准集对CLM进行校准并未提高需要更长推理步骤的任务的一般化能力，这表明在处理更复杂任务方面存在局限性。 

---
# Static Program Analysis Guided LLM Based Unit Test Generation 

**Title (ZH)**: 基于静态程序分析引导的大语言模型单元测试生成 

**Authors**: Sujoy Roychowdhury, Giriprasad Sridhara, A K Raghavan, Joy Bose, Sourav Mazumdar, Hamender Singh, Srinivasan Bajji Sugumaran, Ricardo Britto  

**Link**: [PDF](https://arxiv.org/pdf/2503.05394)  

**Abstract**: We describe a novel approach to automating unit test generation for Java methods using large language models (LLMs). Existing LLM-based approaches rely on sample usage(s) of the method to test (focal method) and/or provide the entire class of the focal method as input prompt and context. The former approach is often not viable due to the lack of sample usages, especially for newly written focal methods. The latter approach does not scale well enough; the bigger the complexity of the focal method and larger associated class, the harder it is to produce adequate test code (due to factors such as exceeding the prompt and context lengths of the underlying LLM). We show that augmenting prompts with \emph{concise} and \emph{precise} context information obtained by program analysis %of the focal method increases the effectiveness of generating unit test code through LLMs. We validate our approach on a large commercial Java project and a popular open-source Java project. 

**Abstract (ZH)**: 我们描述了一种使用大规模语言模型（LLMs）自动生成Java方法单元测试的新方法。现有的基于LLM的方法依赖于方法使用样本（目标方法的示例用法）进行测试，或者提供目标方法的整个类作为输入提示和上下文。前者由于缺乏示例用法，特别是在新编写目标方法的情况下，往往不可行。后者在扩展性方面也存在问题；目标方法及其相关类的复杂度越大，生成足够的测试代码就越困难（由于诸如超出底层LLM提示和上下文长度限制等因素）。我们通过提供由程序分析获得的简洁且精确的上下文信息来增强提示，以提高通过LLM生成单元测试代码的有效性。我们在一个大型商用Java项目和一个流行的开源Java项目上验证了我们的方法。 

---
# Shifting Perspectives: Steering Vector Ensembles for Robust Bias Mitigation in LLMs 

**Title (ZH)**: 转变视角：引导矢量集合在LLMs中实现稳健的偏差减轻 

**Authors**: Zara Siddique, Irtaza Khalid, Liam D. Turner, Luis Espinosa-Anke  

**Link**: [PDF](https://arxiv.org/pdf/2503.05371)  

**Abstract**: We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. We employ Bayesian optimization to systematically identify effective contrastive pair datasets across nine bias axes. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.2%, 4.7%, and 3.2% over the baseline for Mistral, Llama, and Qwen, respectively. Building on these promising results, we introduce Steering Vector Ensembles (SVE), a method that averages multiple individually optimized steering vectors, each targeting a specific bias axis such as age, race, or gender. By leveraging their collective strength, SVE outperforms individual steering vectors in both bias reduction and maintaining model performance. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that SVE is a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety. 

**Abstract (ZH)**: 我们提出了一种通过应用导向矢量来修改大型语言模型（LLMs）前向传递中模型激活的新方法，以减轻偏差。我们使用贝叶斯优化系统地识别出在九个偏差轴上有效的对比 pair 数据集。当在 BBQ 数据集上优化时，我们单独调优的导向矢量分别在 Mistral、Llama 和 Qwen 上实现了基线平均改进 12.2%、4.7% 和 3.2%。基于这些有希望的结果，我们引入了导向矢量集成（SVE）方法，该方法平均多个单独优化的导向矢量，每个导向矢量针对特定的偏差轴（如年龄、种族或性别）。通过利用它们的集体力量，SVE 在降低偏差和保持模型性能方面均优于单独的导向矢量。我们的工作首次系统地研究了导向矢量在减轻偏差方面的应用，并展示了 SVE 是一种强大的且计算效率高的策略，用于减轻 LLM 中的偏差，并具有提高 AI 安全性的广泛意义。 

---
# AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications 

**Title (ZH)**: AutoIOT：由大规模语言模型驱动的自动化自然语言编程 for AIoT 应用 

**Authors**: Leming Shen, Qiang Yang, Yuanqing Zheng, Mo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05346)  

**Abstract**: The advent of Large Language Models (LLMs) has profoundly transformed our lives, revolutionizing interactions with AI and lowering the barrier to AI usage. While LLMs are primarily designed for natural language interaction, the extensive embedded knowledge empowers them to comprehend digital sensor data. This capability enables LLMs to engage with the physical world through IoT sensors and actuators, performing a myriad of AIoT tasks. Consequently, this evolution triggers a paradigm shift in conventional AIoT application development, democratizing its accessibility to all by facilitating the design and development of AIoT applications via natural language. However, some limitations need to be addressed to unlock the full potential of LLMs in AIoT application development. First, existing solutions often require transferring raw sensor data to LLM servers, which raises privacy concerns, incurs high query fees, and is limited by token size. Moreover, the reasoning processes of LLMs are opaque to users, making it difficult to verify the robustness and correctness of inference results. This paper introduces AutoIOT, an LLM-based automated program generator for AIoT applications. AutoIOT enables users to specify their requirements using natural language (input) and automatically synthesizes interpretable programs with documentation (output). AutoIOT automates the iterative optimization to enhance the quality of generated code with minimum user involvement. AutoIOT not only makes the execution of AIoT tasks more explainable but also mitigates privacy concerns and reduces token costs with local execution of synthesized programs. Extensive experiments and user studies demonstrate AutoIOT's remarkable capability in program synthesis for various AIoT tasks. The synthesized programs can match and even outperform some representative baselines. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)的出现深刻改变了我们的生活，通过革新人机互动方式并降低人工智能的应用门槛。虽然LLMs主要用于自然语言交互，但其广泛的内置知识使它们能够理解数字传感器数据。这种能力使LLMs能够通过物联网传感器和执行器与物理世界互动，执行多种AIoT任务。因此，这一演变引发了传统AIoT应用开发范式的转变，通过自然语言简化了AIoT应用的设计和开发。然而，仍需解决一些限制，以充分发挥LLMs在AIoT应用开发中的潜力。首先，现有解决方案通常需要将原始传感器数据传输到LLM服务器，这引发了隐私问题，产生了较高的查询费用，并且受到token数量的限制。此外，LLMs的推理过程对用户不透明，难以验证推断结果的鲁棒性和正确性。本文介绍了AutoIOT，一种基于LLM的AIoT应用自动程序生成器。AutoIOT使用户能够使用自然语言（输入）指定其需求，并自动合成可解释的程序并附带文档（输出）。AutoIOT通过最小化用户参与自动迭代优化，提升生成代码的质量。AutoIOT不仅使AIoT任务的执行更具可解释性，还通过局部执行合成程序来缓解隐私问题和降低token成本。广泛的实验和用户研究证明了AutoIOT在各种AIoT任务中的出色生成程序能力。生成的程序能够与甚至超越一些代表性基准。 

---
# Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models 

**Title (ZH)**: 基于大型语言模型的证据驱动反argument生成中的动态知识集成 

**Authors**: Anar Yeginbergen, Maite Oronoz, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2503.05328)  

**Abstract**: This paper investigates the role of dynamic external knowledge integration in improving counter-argument generation using Large Language Models (LLMs). While LLMs have shown promise in argumentative tasks, their tendency to generate lengthy, potentially unfactual responses highlights the need for more controlled and evidence-based approaches. We introduce a new manually curated dataset of argument and counter-argument pairs specifically designed to balance argumentative complexity with evaluative feasibility. We also propose a new LLM-as-a-Judge evaluation methodology that shows a stronger correlation with human judgments compared to traditional reference-based metrics. Our experimental results demonstrate that integrating dynamic external knowledge from the web significantly improves the quality of generated counter-arguments, particularly in terms of relatedness, persuasiveness, and factuality. The findings suggest that combining LLMs with real-time external knowledge retrieval offers a promising direction for developing more effective and reliable counter-argumentation systems. 

**Abstract (ZH)**: 本论文探究动态外部知识整合在提高大型语言模型（LLMs）反论生成中的作用。虽然LLMs在论辩任务上展现了潜力，但它们生成长篇且可能不实的回应表明需要更加受控和基于证据的方法。我们引入了一个新的手动_curated反论数据集，专门设计以平衡论辩复杂性和评估可行性。我们还提出了一个新的LLM-as-a-Judge评估方法，其与人类判断的相关性比传统引用基标准测量指标更强。实验结果显示，从网络中整合动态外部知识显著提高了生成反论的质量，特别是在相关性、说服力和事实性方面。研究结果表明，将LLMs与实时外部知识检索相结合是开发更有效和可靠反论系统的一个有希望的方向。 

---
# Uncertainty-Aware Decoding with Minimum Bayes Risk 

**Title (ZH)**: 最小贝叶斯风险意识的不确定性解码 

**Authors**: Nico Daheim, Clara Meister, Thomas Möllenhoff, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2503.05318)  

**Abstract**: Despite their outstanding performance in the majority of scenarios, contemporary language models still occasionally generate undesirable outputs, for example, hallucinated text. While such behaviors have previously been linked to uncertainty, there is a notable lack of methods that actively consider uncertainty during text generation. In this work, we show how Minimum Bayes Risk (MBR) decoding, which selects model generations according to an expected risk, can be generalized into a principled uncertainty-aware decoding method. In short, we account for model uncertainty during decoding by incorporating a posterior over model parameters into MBR's computation of expected risk. We show that this modified expected risk is useful for both choosing outputs and deciding when to abstain from generation and can provide improvements without incurring overhead. We benchmark different methods for learning posteriors and show that performance improves with prediction diversity. We release our code publicly. 

**Abstract (ZH)**: 尽管当代语言模型在大多数场景下表现出色，但仍 occasionally生成不 desirable 的输出，例如虚构文本。虽然这些行为之前已与不确定性关联起来，但鲜有方法在文本生成过程中积极考虑不确定性。在本文中，我们展示了如何将最小贝叶斯风险（MBR）解码法推广为一个有原则的不确定性意识解码方法，通过将模型参数的后验概率纳入MBR中期望风险的计算，我们在解码过程中考虑了模型的不确定性。我们证明，这种修改后的期望风险对于选择输出和决定何时停止生成都是有用的，且不会增加额外开销。我们对学习后验的方法进行了基准测试，并显示预测多样性能够提升性能。我们公开发布了我们的代码。 

---
# Knowledge Updating? No More Model Editing! Just Selective Contextual Reasoning 

**Title (ZH)**: 知识更新？不再模型编辑！只需选择性语境推理。 

**Authors**: Guoxiu He, Xin Song, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.05212)  

**Abstract**: As real-world knowledge evolves, the information embedded within large language models (LLMs) can become outdated, inadequate, or erroneous. Model editing has emerged as a prominent approach for updating LLMs' knowledge with minimal computational costs and parameter changes. This approach typically identifies and adjusts specific model parameters associated with newly acquired knowledge. However, existing methods often underestimate the adverse effects that parameter modifications can have on broadly distributed knowledge. More critically, post-edit LLMs frequently struggle with multi-hop reasoning and continuous knowledge updates. Although various studies have discussed these shortcomings, there is a lack of comprehensive evaluation. In this paper, we provide an evaluation of ten model editing methods along four dimensions: reliability, generalization, locality, and portability. Results confirm that all ten popular model editing methods show significant shortcomings across multiple dimensions, suggesting model editing is less promising. We then propose a straightforward method called Selective Contextual Reasoning (SCR), for knowledge updating. SCR does not modify model parameters but harnesses LLM's inherent contextual reasoning capabilities utilizing the updated knowledge pieces. Under SCR, an LLM first assesses whether an incoming query falls within the scope of an external knowledge base. If it does, the relevant external knowledge texts are contextualized to enhance reasoning; otherwise, the query is answered directly. We evaluate SCR against the ten model editing methods on two counterfactual datasets with three backbone LLMs. Empirical results confirm the effectiveness and efficiency of contextual reasoning for knowledge updating. 

**Abstract (ZH)**: 随着现实世界知识的演变，大型语言模型（LLMs）中嵌入的信息可能会变得过时、不足或错误。模型编辑已成为一种突出的方法，用于以最小的计算成本和参数变化更新LLMs的知识。这种方法 typically 通常会识别并调整与新获得知识相关的特定模型参数。然而，现有方法往往低估了参数修改对广泛分布知识的不良影响。更为关键的是，经过编辑的LLMs在多跳推理和连续知识更新方面时常遇到困难。尽管已有多种研究讨论了这些不足之处，但缺乏全面的评估。本文从可靠性、泛化能力、局部性和可移植性四个维度评估了十种模型编辑方法。结果表明，这十种流行的模型编辑方法在多个维度上均显示出显著的不足，建议模型编辑的效果有限。我们随后提出了一个名为Selective Contextual Reasoning（选择性上下文推理，SCR）的简单方法进行知识更新。SCR 不修改模型参数，而是利用LLMs固有的上下文推理能力，结合更新的知识片段。在SCR方法下，LLM首先评估传入查询是否属于外部知识库的范围。如果是，则将相关外部知识文本上下文化以增强推理；否则，直接回答查询。我们使用两个反事实数据集和三种基础LLM对SCR与十种模型编辑方法进行了评估。实验证明，上下文推理在知识更新中的有效性和效率。 

---
# Rewarding Curse: Analyze and Mitigate Reward Modeling Issues for LLM Reasoning 

**Title (ZH)**: 奖励之咒：分析和缓解大语言模型推理中的奖励建模问题 

**Authors**: Jiachun Li, Pengfei Cao, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.05188)  

**Abstract**: Chain-of-thought (CoT) prompting demonstrates varying performance under different reasoning tasks. Previous work attempts to evaluate it but falls short in providing an in-depth analysis of patterns that influence the CoT. In this paper, we study the CoT performance from the perspective of effectiveness and faithfulness. For the former, we identify key factors that influence CoT effectiveness on performance improvement, including problem difficulty, information gain, and information flow. For the latter, we interpret the unfaithful CoT issue by conducting a joint analysis of the information interaction among the question, CoT, and answer. The result demonstrates that, when the LLM predicts answers, it can recall correct information missing in the CoT from the question, leading to the problem. Finally, we propose a novel algorithm to mitigate this issue, in which we recall extra information from the question to enhance the CoT generation and evaluate CoTs based on their information gain. Extensive experiments demonstrate that our approach enhances both the faithfulness and effectiveness of CoT. 

**Abstract (ZH)**: 链式思考（CoT）引导在不同推理任务下的性能表现存在差异。先前的工作尝试对其进行评估，但未能深入分析影响CoT表现的因素模式。本文从有效性与忠实性两个视角研究CoT性能。在有效性方面，我们识别出影响CoT有效性提升的关键因素，包括问题难度、信息增益和信息流。在忠实性方面，我们通过联合分析问题、CoT和答案之间的信息交互来解析不忠实的CoT问题。结果表明，当大模型预测答案时，它可以从问题中召回缺失在CoT中的正确信息，导致问题。最后，我们提出了一种新颖的算法来缓解这一问题，在该算法中，我们从问题中召回额外信息以增强CoT生成，并基于信息增益评价CoT。广泛实验表明，我们的方法能够提升CoT的有效性和忠实性。 

---
# Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching 

**Title (ZH)**: Thought-sketch: 有效的认知启发式素描推理的大型语言模型 

**Authors**: Simon A. Aytes, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.05179)  

**Abstract**: Recent advances in large language models have demonstrated remarkable reasoning capabilities through Chain of Thought (CoT) prompting, but often at the cost of excessive verbosity in their intermediate outputs, which increases computational overhead. We introduce Sketch-of-Thought (SoT), a novel prompting framework that combines cognitive-inspired reasoning paradigms with linguistic constraints to minimize token usage while preserving reasoning accuracy. SoT is designed as a flexible framework that can incorporate any custom reasoning paradigms based on cognitive science, and we instantiate it with three such paradigms - Conceptual Chaining, Chunked Symbolism, and Expert Lexicons - each tailored to different reasoning tasks and selected dynamically via a lightweight routing model. Through comprehensive evaluation across 15 reasoning datasets with multiple languages and multimodal scenarios, we demonstrate that SoT achieves token reductions of 76% with negligible accuracy impact. In certain domains like mathematical and multi-hop reasoning, it even improves accuracy while using significantly fewer tokens. Our code is publicly available: this https URL. 

**Abstract (ZH)**: 最近的大语言模型进展通过Chain of Thought (CoT)提示展示了卓越的推理能力，但常常以中间输出过度冗长为代价，增加了计算开销。我们引入了Sketch-of-Thought (SoT)这一新颖的提示框架，结合认知启发式的推理范式与语言约束，以最小化令牌使用量同时保持推理准确性。SoT设计为一个灵活框架，可以根据认知科学融合任意自定义推理范式，并通过一个轻量级路由模型动态选择三种范式中的任一种——概念链式推理、分块符号主义和专家领域词典，每种范式针对不同的推理任务。通过在15个跨语言和多模态场景的推理数据集中进行全面评估，我们证明SoT在不影响准确性的情况下实现了高达76%的令牌缩减。在某些领域，如数学和多步骤推理中，它甚至在使用显著更少的令牌时提高了准确性。我们的代码已公开：this https URL。 

---
# Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs 

**Title (ZH)**: 每一TFLOP都重要：无需高端GPU即可扩展3000亿参数的专家混合LLM 

**Authors**: Ling Team, Binwei Zeng, Chao Huang, Chao Zhang, Changxin Tian, Cong Chen, Dingnan Jin, Feng Yu, Feng Zhu, Feng Yuan, Fakang Wang, Gangshan Wang, Guangyao Zhai, Haitao Zhang, Huizhong Li, Jun Zhou, Jia Liu, Junpeng Fang, Junjie Ou, Jun Hu, Ji Luo, Ji Zhang, Jian Liu, Jian Sha, Jianxue Qian, Jiewei Wu, Junping Zhao, Jianguo Li, Jubao Feng, Jingchao Di, Junming Xu, Jinghua Yao, Kuan Xu, Kewei Du, Longfei Li, Lei Liang, Lu Yu, Li Tang, Lin Ju, Peng Xu, Qing Cui, Song Liu, Shicheng Li, Shun Song, Song Yan, Tengwei Cai, Tianyi Chen, Ting Guo, Ting Huang, Tao Feng, Tao Wu, Wei Wu, Xiaolu Zhang, Xueming Yang, Xin Zhao, Xiaobo Hu, Xin Lin, Yao Zhao, Yilong Wang, Yongzhen Guo, Yuanyuan Wang, Yue Yang, Yang Cao, Yuhao Fu, Yi Xiong, Yanzhe Li, Zhe Li, Zhiqiang Zhang, Ziqi Liu, Zhaoxin Huan, Zujie Wen, Zhenhang Sun, Zhuoxuan Du, Zhengyu He  

**Link**: [PDF](https://arxiv.org/pdf/2503.05139)  

**Abstract**: In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled Bǎilíng in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at this https URL. 

**Abstract (ZH)**: 本技术报告探讨了训练大规模专家混合模型（MoE）的挑战，重点在于克服此类系统中普遍存在的成本不效率和资源限制。为了解决这些问题，我们展示了两种不同规模的MoE大型语言模型（LLMs），分别是Ling-Lite和Ling-Plus（中文简称“摆灵”，拼音Bǎilíng）。Ling-Lite包含168亿参数，其中激活参数为2.75亿，而Ling-Plus则包含2900亿参数，激活参数为288亿。两者在性能上均与行业领先的标准相当。本报告提供了在资源受限环境中提高AI开发效率和可访问性的可行建议，促进更具扩展性和可持续性的技术。具体而言，为了减少大规模MoE模型的训练成本，我们提出了优化模型架构和训练过程、改进训练异常处理以及提高模型评估效率的创新方法。此外，利用从知识图谱中生成的高质量数据，我们的模型在工具使用能力上优于其他模型。最终，我们的实验结果表明，一个300亿参数的MoE LLM可以在较低性能的设备上有效训练，并达到与类似规模模型相当的性能，包括密集型和MoE模型。与高性能设备相比，在预训练阶段使用较低配置的硬件系统可以节省显著的计算成本，降低约20%的计算成本。这些模型可通过以下链接访问：这个 https URL。 

---
# Multi-Task Reinforcement Learning Enables Parameter Scaling 

**Title (ZH)**: 多任务强化学习实现参数缩放 

**Authors**: Reginald McLean, Evangelos Chataroulas, Jordan Terry, Isaac Woungang, Nariman Farsad, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2503.05126)  

**Abstract**: Multi-task reinforcement learning (MTRL) aims to endow a single agent with the ability to perform well on multiple tasks. Recent works have focused on developing novel sophisticated architectures to improve performance, often resulting in larger models; it is unclear, however, whether the performance gains are a consequence of the architecture design itself or the extra parameters. We argue that gains are mostly due to scale by demonstrating that naively scaling up a simple MTRL baseline to match parameter counts outperforms the more sophisticated architectures, and these gains benefit most from scaling the critic over the actor. Additionally, we explore the training stability advantages that come with task diversity, demonstrating that increasing the number of tasks can help mitigate plasticity loss. Our findings suggest that MTRL's simultaneous training across multiple tasks provides a natural framework for beneficial parameter scaling in reinforcement learning, challenging the need for complex architectural innovations. 

**Abstract (ZH)**: 多任务强化学习（MTRL）旨在赋予单个智能体在多个任务上表现出色的能力。近期研究重点在于开发新颖复杂的架构以提高性能，通常会导致模型规模增大；然而，性能提升的原因是由于架构设计本身还是额外的参数尚不明确。我们 argue 认为增益主要源于规模的扩大，通过展示简单 MTRL 基线模型盲目扩大以匹配参数数量的表现超过了更为复杂的架构，并且这些增益主要得益于对价值函数而非策略函数的规模扩大。此外，我们还探讨了任务多样性带来的训练稳定性优势，证明增加任务数量有助于减轻模型适应性损失。我们的发现表明，MTRL 跨多个任务的同时训练为强化学习中的有益参数规模扩大提供了一个自然框架，挑战了复杂架构创新的必要性。 

---
# PromptPex: Automatic Test Generation for Language Model Prompts 

**Title (ZH)**: PromptPex：自动生成语言模型提示的测试用例 

**Authors**: Reshabh K Sharma, Jonathan De Halleux, Shraddha Barke, Benjamin Zorn  

**Link**: [PDF](https://arxiv.org/pdf/2503.05070)  

**Abstract**: Large language models (LLMs) are being used in many applications and prompts for these models are integrated into software applications as code-like artifacts. These prompts behave much like traditional software in that they take inputs, generate outputs, and perform some specific function. However, prompts differ from traditional code in many ways and require new approaches to ensure that they are robust. For example, unlike traditional software the output of a prompt depends on the AI model that interprets it. Also, while natural language prompts are easy to modify, the impact of updates is harder to predict. New approaches to testing, debugging, and modifying prompts with respect to the model running them are required.
To address some of these issues, we developed PromptPex, an LLM-based tool to automatically generate and evaluate unit tests for a given prompt. PromptPex extracts input and output specifications from a prompt and uses them to generate diverse, targeted, and valid unit tests. These tests are instrumental in identifying regressions when a prompt is changed and also serve as a tool to understand how prompts are interpreted by different models. We use PromptPex to generate tests for eight benchmark prompts and evaluate the quality of the generated tests by seeing if they can cause each of four diverse models to produce invalid output. PromptPex consistently creates tests that result in more invalid model outputs than a carefully constructed baseline LLM-based test generator. Furthermore, by extracting concrete specifications from the input prompt, PromptPex allows prompt writers to clearly understand and test specific aspects of their prompts. The source code of PromptPex is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种应用中被使用，这些模型的提示作为代码-like制品集成到软件应用中。这些提示在许多方面类似于传统软件，它们接受输入、生成输出并执行特定功能。然而，提示与传统代码有许多不同之处，需要采取新的方法以确保其稳健性。例如，与传统软件不同，提示的输出取决于解释它的AI模型。此外，虽然自然语言提示容易修改，但更新的影响更难预测。对提示与运行它们的模型相关的测试、调试和修改方法需要新的方法。

为解决部分问题，我们开发了PromptPex，这是一种基于LLM的工具，用于自动为给定的提示生成和评估单元测试。PromptPex从提示中提取输入和输出规范，并使用它们生成多样、针对性且有效的单元测试。这些测试对于提示更改时识别回归至关重要，同时也作为工具来理解不同模型如何解释提示。我们使用PromptPex为八个基准提示生成测试，并通过评估生成测试能否导致四个不同模型生成无效输出来衡量测试质量。PromptPex生成的测试始终导致比精心构建的基本LLM测试生成器更多的无效模型输出。此外，通过从输入提示中提取具体的规范，PromptPex允许提示编写者清晰地理解和测试其提示的具体方面。PromptPex的源代码可从此链接获取。 

---
# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts 

**Title (ZH)**: 容量感知推断：减轻混合专家模型中的游荡者效应 

**Authors**: Shwai He, Weilin Cai, Jiayi Huang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.05066)  

**Abstract**: The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation, optimizing the trade-off between performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where some experts are overloaded while others remain underutilized. This imbalance leads to poor resource utilization and increased latency, as the most burdened expert dictates the overall delay, a phenomenon we define as the \textbf{\textit{Straggler Effect}}. To mitigate this, we propose Capacity-Aware Inference, including two key techniques: (1) \textbf{\textit{Capacity-Aware Token Drop}}, which discards overloaded tokens to regulate the maximum latency of MoE, and (2) \textbf{\textit{Capacity-Aware Token Reroute}}, which reallocates overflowed tokens to underutilized experts, balancing the token distribution. These techniques collectively optimize both high-load and low-load expert utilization, leading to a more efficient MoE inference pipeline. Extensive experiments demonstrate the effectiveness of our methods, showing significant improvements in inference efficiency, e.g., 0.2\% average performance increase and a 1.94$\times$ inference speedup on Mixtral-8$\times$7B-Instruct. 

**Abstract (ZH)**: 专家混合模型（MoE）的有效扩展架构通过利用稀疏专家激活来扩展大规模语言模型，优化性能与效率之间的 trade-off。然而，在专家并行处理下，MoE 因 token-to-expert 分配不平衡而导致推理效率低下，一些专家超载而另一些则利用不足。这种不平衡导致资源利用效率低下和延迟增加，我们将其现象定义为“拖后腿效应”（Straggler Effect）。为缓解这一问题，我们提出了容量感知推理，包括两种关键技术：（1）容量感知 token 筛选（Capacity-Aware Token Drop），通过丢弃超载 token 来调节 MoE 的最大延迟；（2）容量感知 token 重分配（Capacity-Aware Token Reroute），通过将 overflowed token 重新分配给利用不足的专家，平衡 token 分布。这些技术共同优化高负载和低负载专家的利用，从而实现更高效的 MoE 推理管道。大量实验证明了我们方法的有效性，显示出推理效率的显著提高，例如平均性能提高 0.2%，Mixedral-8×7B-Instruct 上推理速度提升 1.94 倍。 

---
# Continual Pre-training of MoEs: How robust is your router? 

**Title (ZH)**: MoEs的持续预训练：你的路由器有多 robust？ 

**Authors**: Benjamin Thérien, Charles-Étienne Joseph, Zain Sarwar, Ashwinee Panda, Anirban Das, Shi-Xiong Zhang, Stephen Rawls, Sambit Sahu, Eugene Belilovsky, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2503.05029)  

**Abstract**: Sparsely-activated Mixture of Experts (MoE) transformers are promising architectures for foundation models. Compared to dense transformers that require the same amount of floating point operations (FLOPs) per forward pass, MoEs benefit from improved sample efficiency at training time and achieve much stronger performance. Many closed-source and open-source frontier language models have thus adopted an MoE architecture. Naturally, practitioners will want to extend the capabilities of these models with large amounts of newly collected data without completely re-training them. Prior work has shown that a simple combination of replay and learning rate re-warming and re-decaying can enable the continual pre-training (CPT) of dense decoder-only transformers with minimal performance degradation compared to full re-training. In the case of decoder-only MoE transformers, however, it is unclear how the routing algorithm will impact continual pre-training performance: 1) do the MoE transformer's routers exacerbate forgetting relative to a dense model?; 2) do the routers maintain a balanced load on previous distributions after CPT?; 3) are the same strategies applied to dense models sufficient to continually pre-train MoE LLMs? In what follows, we conduct a large-scale (>2B parameter switch and DeepSeek MoE LLMs trained for 600B tokens) empirical study across four MoE transformers to answer these questions. Our results establish a surprising robustness to distribution shifts for both Sinkhorn-Balanced and Z-and-Aux-loss-balanced routing algorithms, even in MoEs continually pre-trained without replay. Moreover, we show that MoE LLMs maintain their sample efficiency (relative to a FLOP-matched dense model) during CPT and that they can match the performance of a fully re-trained MoE at a fraction of the cost. 

**Abstract (ZH)**: 稀疏激活 experts 混合的变换器架构（MoE）是基础模型的有前途的架构。与需要每次前向传递相同浮点运算（FLOPs）数量的密集变换器相比，MoE 在训练时受益于改进的样本效率并实现更强大的性能。因此，许多闭源和开源前沿语言模型采用了 MoE 架构。自然地，实践者希望利用大量新收集的数据扩展这些模型的能力，而无需完全重新训练它们。以往的工作表明，简单的回放与学习率重新温暖和衰减相结合可以使得密集的解码器-only 变换器的持续预训练（CPT）在与完全重新训练相比的最小性能下降下得以实现。然而，在解码器-only MoE 变换器的情况下，路由算法对持续预训练性能的影响尚不清楚：1）MoE 变换器的路由器是否相对于密集模型加剧了遗忘？2）路由器在持续预训练后是否能够保持对以前分布的平衡负担？3）应用于密集模型的策略是否足够用于持续预训练 MoE 大型语言模型？我们随后通过四类 MoE 变换器进行的一项大规模（超过 2 亿参数的切换和 DeepSeek MoE 大型语言模型，训练了 600 亿个标记）实证研究来回答这些问题。我们的结果表明，即使在没有回放的情况下持续预训练 MoE 中，Sinkhorn-Balanced 和 Z-and-Aux-loss-balanced 路由算法也表现出惊人的鲁棒性。此外，我们证明，在持续预训练中，MoE 大型语言模型保持了与 FLOP 匹配的密集模型相当的样本效率，并且它们可以在极低的成本下匹配完全重新训练的 MoE 的性能。 

---
# LLMs' Reshaping of People, Processes, Products, and Society in Software Development: A Comprehensive Exploration with Early Adopters 

**Title (ZH)**: LLMs对软件开发中的人、流程、产品和社会的重塑：早期采用者的全面探索 

**Authors**: Benyamin Tabarsi, Heidi Reichert, Ally Limke, Sandeep Kuttal, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2503.05012)  

**Abstract**: Large language models (LLMs) like OpenAI ChatGPT, Google Gemini, and GitHub Copilot are rapidly gaining traction in the software industry, but their full impact on software engineering remains insufficiently explored. Despite their growing adoption, there is a notable lack of formal, qualitative assessments of how LLMs are applied in real-world software development contexts. To fill this gap, we conducted semi-structured interviews with sixteen early-adopter professional developers to explore their use of LLMs throughout various stages of the software development life cycle. Our investigation examines four dimensions: people - how LLMs affect individual developers and teams; process - how LLMs alter software engineering workflows; product - LLM impact on software quality and innovation; and society - the broader socioeconomic and ethical implications of LLM adoption. Thematic analysis of our data reveals that while LLMs have not fundamentally revolutionized the development process, they have substantially enhanced routine coding tasks, including code generation, refactoring, and debugging. Developers reported the most effective outcomes when providing LLMs with clear, well-defined problem statements, indicating that LLMs excel with decomposed problems and specific requirements. Furthermore, these early-adopters identified that LLMs offer significant value for personal and professional development, aiding in learning new languages and concepts. Early-adopters, highly skilled in software engineering and how LLMs work, identified early and persisting challenges for software engineering, such as inaccuracies in generated content and the need for careful manual review before integrating LLM outputs into production environments. Our study provides a nuanced understanding of how LLMs are shaping the landscape of software development, with their benefits, limitations, and ongoing implications. 

**Abstract (ZH)**: 大型语言模型（LLMs）如OpenAI ChatGPT、Google Gemini和GitHub Copilot在软件行业中的应用正迅速增长，但其对软件工程的全面影响尚未充分探讨。尽管其采用率不断增加，但对LLMs在实际软件开发环境中的应用进行了正式和定性评估的情况仍然较少。为填补这一空白，我们对十六名早期采用者的专业开发人员进行了半结构化访谈，以探索他们在软件开发生命周期各阶段使用LLMs的情况。我们的研究考察了四个维度：人——LLMs如何影响个体开发人员和团队；过程——LLMs如何改变软件工程工作流程；产品——LLMs对软件质量和创新的影响；社会——LLMs采用的更广泛的经济社会和伦理影响。对我们的数据进行主题分析揭示了以下内容：虽然LLMs尚未根本改变开发流程，但它们极大地提高了常规编码任务的效率，包括代码生成、重构和调试。开发人员表示，当提供给LLMs清晰明确的问题陈述时，其效果最佳，表明LLMs在分解问题和明确需求方面表现出色。此外，早期采用者认为LLMs在个人和职业发展中提供了重大价值，有助于学习新的语言和概念。早期采用者，软件工程和LLMs工作的高手，指出了软件工程领域早期和持续存在的挑战，如生成内容的不准确性以及在将LLM输出集成到生产环境之前需要进行仔细的手动审查的必要性。本研究提供了对LLMs如何塑造软件开发格局的细致理解，包括其优点、局限性和持续影响。 

---
# Balcony: A Lightweight Approach to Dynamic Inference of Generative Language Models 

**Title (ZH)**: 阳台：生成语言模型动态推理的一种轻量级方法 

**Authors**: Benyamin Jamialahmadi, Parsa Kavehzadeh, Mehdi Rezagholizadeh, Parsa Farinneya, Hossein Rajabzadeh, Aref Jafari, Boxing Chen, Marzieh Tahaei  

**Link**: [PDF](https://arxiv.org/pdf/2503.05005)  

**Abstract**: Deploying large language models (LLMs) in real-world applications is often hindered by strict computational and latency constraints. While dynamic inference offers the flexibility to adjust model behavior based on varying resource budgets, existing methods are frequently limited by hardware inefficiencies or performance degradation. In this paper, we introduce Balcony, a simple yet highly effective framework for depth-based dynamic inference. By freezing the pretrained LLM and inserting additional transformer layers at selected exit points, Balcony maintains the full model's performance while enabling real-time adaptation to different computational budgets. These additional layers are trained using a straightforward self-distillation loss, aligning the sub-model outputs with those of the full model. This approach requires significantly fewer training tokens and tunable parameters, drastically reducing computational costs compared to prior methods. When applied to the LLaMA3-8B model, using only 0.2% of the original pretraining data, Balcony achieves minimal performance degradation while enabling significant speedups. Remarkably, we show that Balcony outperforms state-of-the-art methods such as Flextron and Layerskip as well as other leading compression techniques on multiple models and at various scales, across a variety of benchmarks. 

**Abstract (ZH)**: 部署大型语言模型（LLMs）在实际应用中往往受限于严格的计算和延迟约束。虽然动态推理可以根据不同的资源预算调整模型行为，但现有方法常常受到硬件效率低下或性能下降的限制。本文介绍了一种基于深度的简单而高效的动态推理框架——Balcony。通过冻结预训练的LLM并在选定的退出点插入额外的Transformer层，Balcony能够在保持完整模型性能的同时，实现对不同计算预算的即时适应。这些额外的层是通过一个简单的自蒸馏损失进行训练的，使得子模型输出与完整模型的输出相一致。这种方法所需的训练令牌和可调参数数量显著减少，相比以往方法大幅降低了计算成本。当应用于LLaMA3-8B模型时，仅使用原始预训练数据的0.2%，Balcony仍能实现性能微小下降的同时带来显著的加速。令人惊讶的是，我们证明了Balcony在多种模型和不同规模下，在多个基准测试中优于Flextron、Layerskip等最新方法以及其他领先的压缩技术。 

---
# Wanda++: Pruning Large Language Models via Regional Gradients 

**Title (ZH)**: Wanda++: 基于区域梯度裁剪大型语言模型 

**Authors**: Yifan Yang, Kai Zhen, Bhavana Ganesh, Aram Galstyan, Goeric Huybrechts, Markus Müller, Jonas M. Kübler, Rupak Vignesh Swaminathan, Athanasios Mouchtaris, Sravan Babu Bodapati, Nathan Susanj, Zheng Zhang, Jack FitzGerald, Abhishek Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2503.04992)  

**Abstract**: Large Language Models (LLMs) pruning seeks to remove unimportant weights for inference speedup with minimal performance impact. However, existing methods often suffer from performance loss without full-model sparsity-aware fine-tuning. This paper presents Wanda++, a novel pruning framework that outperforms the state-of-the-art methods by utilizing decoder-block-level \textbf{regional} gradients. Specifically, Wanda++ improves the pruning score with regional gradients for the first time and proposes an efficient regional optimization method to minimize pruning-induced output discrepancies between the dense and sparse decoder output. Notably, Wanda++ improves perplexity by up to 32\% over Wanda in the language modeling task and generalizes effectively to downstream tasks. Further experiments indicate our proposed method is orthogonal to sparsity-aware fine-tuning, where Wanda++ can be combined with LoRA fine-tuning to achieve a similar perplexity improvement as the Wanda method. The proposed method is lightweight, pruning a 7B LLaMA model in under 10 minutes on a single NVIDIA H100 GPU. 

**Abstract (ZH)**: 大规模语言模型（LLMs）剪枝旨在通过移除不重要权重来提高推理速度，同时最小化性能影响。然而，现有方法往往在未进行全模型稀疏意识微调的情况下会遭受性能下降。本文提出了一种新型剪枝框架Wanda++，该框架通过利用解码块级区域梯度超越了现有最佳方法。具体而言，Wanda++首次利用区域梯度来改进剪枝分数，并提出了一种高效的区域优化方法，以最小化剪枝引起的稠密和稀疏解码器输出之间的输出差异。值得注意的是，Wanda++在语言建模任务中将困惑度提高了高达32%，并且在下游任务上表现出了有效的泛化能力。进一步的实验表明，我们提出的方法与稀疏意识微调正交，Wanda++可以与LoRA微调结合使用，以达到与Wanda方法相似的困惑度改进效果。所提出的方法轻量级，在单块NVIDIA H100 GPU上对一个7B LLaMA模型进行剪枝不到10分钟。 

---
# LVLM-Compress-Bench: Benchmarking the Broader Impact of Large Vision-Language Model Compression 

**Title (ZH)**: LVLM-Compress-Bench: 评估大规模视觉-语言模型压缩的更广泛影响 

**Authors**: Souvik Kundu, Anahita Bhiwandiwalla, Sungduk Yu, Phillip Howard, Tiep Le, Sharath Nittur Sridhar, David Cobbley, Hao Kang, Vasudev Lal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04982)  

**Abstract**: Despite recent efforts in understanding the compression impact on large language models (LLMs) in terms of their downstream task performance and trustworthiness on relatively simpler uni-modal benchmarks (for example, question answering, common sense reasoning), their detailed study on multi-modal Large Vision-Language Models (LVLMs) is yet to be unveiled. Towards mitigating this gap, we present LVLM-Compress-Bench, a framework to first thoroughly study the broad impact of compression on the generative performance of LVLMs with multi-modal input driven tasks. In specific, we consider two major classes of compression for autoregressive models, namely KV cache and weight compression, for the dynamically growing intermediate cache and static weights, respectively.
We use four LVLM variants of the popular LLaVA framework to present our analysis via integrating various state-of-the-art KV and weight compression methods including uniform, outlier-reduced, and group quantization for the KV cache and weights. With this framework we demonstrate on ten different multi-modal datasets with different capabilities including recognition, knowledge, language generation, spatial awareness, visual reasoning, hallucination and visual illusion identification, toxicity, stereotypes and bias. In specific, our framework demonstrates the compression impact on both general and ethically critical metrics leveraging a combination of real world and synthetic datasets to encompass diverse societal intersectional attributes. Extensive experimental evaluations yield diverse and intriguing observations on the behavior of LVLMs at different quantization budget of KV and weights, in both maintaining and losing performance as compared to the baseline model with FP16 data format.
Code will be open-sourced at this https URL. 

**Abstract (ZH)**: 尽管近年来在理解压缩对大型语言模型（LLMs）的影响方面取得了一定进展，特别是在其下游任务性能和相对简单的单模基准上的可信度方面（例如，问答、常识推理），但对于多模态大型视觉-语言模型（LVLMs）的详细研究仍然尚未展开。为弥补这一差距，我们提出了LVLM-Compress-Bench框架，旨在全面研究压缩对多模态输入驱动任务中LVLMs生成性能的影响。具体而言，我们考虑了两类自回归模型的压缩方法，分别是用于动态增长的中间缓存的KV缓存压缩和用于静态权重的权重压缩。

我们使用流行的LLaVA框架的四种LVLM变体，通过集成最新的KV和权重压缩方法（包括均匀量化、异常值减少量化和分组量化）来展示我们的分析。通过该框架，我们在包括识别、知识、语言生成、空间意识、视觉推理、幻觉和视觉错觉识别、毒性、刻板印象和偏见等多种能力的十个不同多模态数据集上进行了演示。具体而言，我们的框架利用结合现实世界和合成数据集来展示压缩对通用性和伦理关键性指标的影响，这些数据集涵盖了多样化的社会交叉属性。广泛的实验评估揭示了在不同量化预算下的KV和权重压缩对LVLMs行为的影响，在保持和损失性能方面与FP16数据格式的基线模型相比呈现出多样而有趣的观察结果。

代码将在此处开放源代码：this https URL。 

---
# Beyond RAG: Task-Aware KV Cache Compression for Comprehensive Knowledge Reasoning 

**Title (ZH)**: 超越RAG：面向任务的键值缓存压缩以实现全面知识推理 

**Authors**: Giulio Corallo, Orion Weller, Fabio Petroni, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2503.04973)  

**Abstract**: Incorporating external knowledge in large language models (LLMs) enhances their utility across diverse applications, but existing methods have trade-offs. Retrieval-Augmented Generation (RAG) fetches evidence via similarity search, but key information may fall outside top ranked results. Long-context models can process multiple documents but are computationally expensive and limited by context window size. Inspired by students condensing study material for open-book exams, we propose task-aware key-value (KV) cache compression, which compresses external knowledge in a zero- or few-shot setup. This enables LLMs to reason efficiently over a compacted representation of all relevant information. Experiments show our approach outperforms both RAG and task-agnostic compression methods. On LongBench v2, it improves accuracy by up to 7 absolute points over RAG with a 30x compression rate, while reducing inference latency from 0.43s to 0.16s. A synthetic dataset highlights that RAG performs well when sparse evidence suffices, whereas task-aware compression is superior for broad knowledge tasks. 

**Abstract (ZH)**: 在大型语言模型中整合外部知识可以增强其在多种应用中的实用性，但现有方法存在权衡。受学生为开卷考试浓缩学习资料的启发，我们提出了一种任务感知的关键值缓存压缩方法，该方法在零样本或少样本设置中压缩外部知识，使大型语言模型能够高效地推理所有相关信息的紧凑表示。实验表明，我们的方法在性能上优于检索增强生成（RAG）和任务无关的压缩方法。在LongBench v2上，使用30倍压缩率时，准确率提高了7个绝对点，并将推理延迟从0.43秒减少到0.16秒。合成数据集表明，当证据稀疏时RAG表现良好，而任务感知压缩对于广泛知识任务更优。 

---
# SafeArena: Evaluating the Safety of Autonomous Web Agents 

**Title (ZH)**: SafeArena: 评估自主网络代理的安全性 

**Authors**: Ada Defne Tur, Nicholas Meade, Xing Han Lù, Alejandra Zambrano, Arkil Patel, Esin Durmus, Spandana Gella, Karolina Stańczak, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.04957)  

**Abstract**: LLM-based agents are becoming increasingly proficient at solving web-based tasks. With this capability comes a greater risk of misuse for malicious purposes, such as posting misinformation in an online forum or selling illicit substances on a website. To evaluate these risks, we propose SafeArena, the first benchmark to focus on the deliberate misuse of web agents. SafeArena comprises 250 safe and 250 harmful tasks across four websites. We classify the harmful tasks into five harm categories -- misinformation, illegal activity, harassment, cybercrime, and social bias, designed to assess realistic misuses of web agents. We evaluate leading LLM-based web agents, including GPT-4o, Claude-3.5 Sonnet, Qwen-2-VL 72B, and Llama-3.2 90B, on our benchmark. To systematically assess their susceptibility to harmful tasks, we introduce the Agent Risk Assessment framework that categorizes agent behavior across four risk levels. We find agents are surprisingly compliant with malicious requests, with GPT-4o and Qwen-2 completing 34.7% and 27.3% of harmful requests, respectively. Our findings highlight the urgent need for safety alignment procedures for web agents. Our benchmark is available here: this https URL 

**Abstract (ZH)**: 基于LLM的代理在解决网络任务方面的能力日益增强。随着这一能力的提升，代理被恶意利用的风险也在增加，比如在在线论坛发布虚假信息或在网上销售非法物品。为评估这些风险，我们提出了SafeArena，这是首个专注于网络代理恶意利用行为的标准基准。SafeArena包含四个网站上的250个安全任务和250个有害任务。我们将有害任务划分为五类危害类别——虚假信息、非法活动、骚扰、网络犯罪和社会偏见，旨在评估网络代理的现实滥用情况。我们对包括GPT-4o、Claude-3.5 Sonnet、Qwen-2-VL 72B和Llama-3.2 90B在内的领先基于LLM的网络代理进行评估。为了系统地评估其对有害任务的易感性，我们引入了代理风险评估框架，该框架将代理行为分为四个风险等级。我们发现，代理竟然相当合规地执行了恶意请求，GPT-4o和Qwen-2分别完成了34.7%和27.3%的有害请求。我们的研究结果突显了迫切需要对网络代理进行安全对齐程序。我们的基准数据可在此获取：this https URL 

---
# Are Large Language Models Good In-context Learners for Financial Sentiment Analysis? 

**Title (ZH)**: 大型语言模型在金融情感分析中的即插即用学习能力良好吗？ 

**Authors**: Xinyu Wei, Luojia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04873)  

**Abstract**: Recently, large language models (LLMs) with hundreds of billions of parameters have demonstrated the emergent ability, surpassing traditional methods in various domains even without fine-tuning over domain-specific data. However, when it comes to financial sentiment analysis (FSA)$\unicode{x2013}$a fundamental task in financial AI$\unicode{x2013}$these models often encounter various challenges, such as complex financial terminology, subjective human emotions, and ambiguous inclination expressions. In this paper, we aim to answer the fundamental question: whether LLMs are good in-context learners for FSA? Unveiling this question can yield informative insights on whether LLMs can learn to address the challenges by generalizing in-context demonstrations of financial document-sentiment pairs to the sentiment analysis of new documents, given that finetuning these models on finance-specific data is difficult, if not impossible at all. To the best of our knowledge, this is the first paper exploring in-context learning for FSA that covers most modern LLMs (recently released DeepSeek V3 included) and multiple in-context sample selection methods. Comprehensive experiments validate the in-context learning capability of LLMs for FSA. 

**Abstract (ZH)**: 近期，具有数百亿参数的大型语言模型（LLMs）在各种领域展现了超越传统方法的能力，即使在没有针对特定领域进行微调的情况下也是如此。然而，在金融情感分析（FSA）这一金融AI的基本任务中，这些模型常常会遇到诸如复杂的金融术语、主观的人类情感以及模糊的倾向性表达等挑战。本文旨在回答一个基本问题：LLMs 是否适合用于FSA的上下文学习？揭开这个问题的答案可以揭示LLMs是否能够通过泛化金融文档-情感对的上下文示例来解决FSA的新文档情感分析问题，特别是在难以甚至不可能针对金融特定数据进行模型微调的情况下。据我们所知，这是第一篇探讨涵盖大多数现代LLM（包括最近发布的DeepSeek V3）和多种上下文样本选择方法的FSA上下文学习的论文。全面的实验验证了LLMs在FSA上的上下文学习能力。 

---
# TinyR1-32B-Preview: Boosting Accuracy with Branch-Merge Distillation 

**Title (ZH)**: TinyR1-32B-预览：基于分支合并蒸馏提升准确性 

**Authors**: Lin Sun, Guangxiang Zhao, Xiaoqi Jian, Yuhan Wu, Weihong Lin, Yongfu Zhu, Change Jia, Linglin Zhang, Jinzhu Wu, Junfeng Ran, Sai-er Hu, Zihan Jiang, Junting Zhou, Wenrui Liu, Bin Cui, Tong Yang, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.04872)  

**Abstract**: The challenge of reducing the size of Large Language Models (LLMs) while maintaining their performance has gained significant attention. However, existing methods, such as model distillation and transfer learning, often fail to achieve high accuracy. To address this limitation, we introduce the Branch-Merge distillation approach, which enhances model compression through two phases: (1) the Branch Phase, where knowledge from a large teacher model is \textit{selectively distilled} into specialized student models via domain-specific supervised fine-tuning (SFT); And (2) the Merge Phase, where these student models are merged to enable cross-domain knowledge transfer and improve generalization. We validate our distillation approach using DeepSeek-R1 as the teacher and DeepSeek-R1-Distill-Qwen-32B as the student. The resulting merged model, TinyR1-32B-Preview, outperforms its counterpart DeepSeek-R1-Distill-Qwen-32B across multiple benchmarks, including Mathematics (+5.5 points), Coding (+4.4 points) and Science (+2.9 points), while achieving near-equal performance to DeepSeek-R1 on AIME 2024. The Branch-Merge distillation approach provides a scalable solution for creating smaller, high-performing LLMs with reduced computational cost and time. 

**Abstract (ZH)**: 减小大型语言模型规模以维持其性能的支分合并蒸馏方法面临着重大挑战 

---
# Codebook Reduction and Saturation: Novel observations on Inductive Thematic Saturation for Large Language Models and initial coding in Thematic Analysis 

**Title (ZH)**: 代码本缩减与饱和：大型语言模型归纳主题饱和的新观察及其在主题分析初步编码中的应用 

**Authors**: Stefano De Paoli, Walter Stan Mathis  

**Link**: [PDF](https://arxiv.org/pdf/2503.04859)  

**Abstract**: This paper reflects on the process of performing Thematic Analysis with Large Language Models (LLMs). Specifically, the paper deals with the problem of analytical saturation of initial codes, as produced by LLMs. Thematic Analysis is a well-established qualitative analysis method composed of interlinked phases. A key phase is the initial coding, where the analysts assign labels to discrete components of a dataset. Saturation is a way to measure the validity of a qualitative analysis and relates to the recurrence and repetition of initial codes. In the paper we reflect on how well LLMs achieve analytical saturation and propose also a novel technique to measure Inductive Thematic Saturation (ITS). This novel technique leverages a programming framework called DSPy. The proposed novel approach allows a precise measurement of ITS. 

**Abstract (ZH)**: 本研究反思了使用大规模语言模型（LLMs）进行主题分析的过程，特别关注初始编码的分析饱和问题。主题分析是一种成熟的定性分析方法，由多个相互关联的阶段组成。初始编码是一个关键阶段，分析师为数据集中的离散组件分配标签。饱和度是一种衡量定性分析有效性的方法，与初始编码的重复和重现有关。本研究反思了LLMs实现分析饱和的效果，并提出了一种新的技术来衡量归纳主题饱和（ITS）。该新技术利用了名为DSPy的编程框架，提出的新型方法能够精确测量ITS。 

---
# SHAPE : Self-Improved Visual Preference Alignment by Iteratively Generating Holistic Winner 

**Title (ZH)**: SHAPE : 自我改进的视觉偏好对齐通过迭代生成全局胜者 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Jiazhen Yang, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.04858)  

**Abstract**: Large Visual Language Models (LVLMs) increasingly rely on preference alignment to ensure reliability, which steers the model behavior via preference fine-tuning on preference data structured as ``image - winner text - loser text'' triplets. However, existing approaches often suffer from limited diversity and high costs associated with human-annotated preference data, hindering LVLMs from fully achieving their intended alignment capabilities. We present \projectname, a self-supervised framework capable of transforming the already abundant supervised text-image pairs into holistic preference triplets for more effective and cheaper LVLM alignment, eliminating the need for human preference annotations. Our approach facilitates LVLMs in progressively enhancing alignment capabilities through iterative self-improvement. The key design rationale is to devise preference triplets where the winner text consistently improves in holisticness and outperforms the loser response in quality, thereby pushing the model to ``strive to the utmost'' of alignment performance through preference fine-tuning. For each given text-image pair, SHAPE introduces multiple visual augmentations and pairs them with a summarized text to serve as the winner response, while designating the original text as the loser response. Experiments across \textbf{12} benchmarks on various model architectures and sizes, including LLaVA and DeepSeek-VL, show that SHAPE achieves significant gains, for example, achieving +11.3\% on MMVet (comprehensive evaluation), +1.4\% on MMBench (general VQA), and +8.0\% on POPE (hallucination robustness) over baselines in 7B models. Notably, qualitative analyses confirm enhanced attention to visual details and better alignment with human preferences for holistic descriptions. 

**Abstract (ZH)**: Large Visual Language Models (LVLMs) 越来越多地依赖偏好对齐以确保可靠性，这通过偏好微调引导模型行为，偏好数据结构化为“图像-获胜文本-失败文本”三元组。然而，现有方法通常受到有限多样性和与人类标注偏好数据相关的高成本的限制，阻碍了 LVLMs 完全实现其预期的对齐能力。我们提出 \projectname，一个自监督框架，能够将已经丰富的监督文本-图像配对转换为整体偏好三元组，以实现更有效的且成本更低的 LVLM 对齐，从而消除对人类偏好标注的需求。我们的方法通过逐步自我改进帮助 LVLMs 提高对齐能力。核心设计原则是设计偏好三元组，在这些三元组中，获胜文本在整体性上持续改进，并在质量上优于失败响应，从而通过偏好微调促使模型“尽最大努力”提高对齐性能。对于每个给定的文本-图像配对，SHAPE 引入多种视觉增强并将其与总结的文本配对，作为获胜响应，而将原始文本指定为失败响应。在包括 LLaVA 和 DeepSeek-VL 在内的各种模型架构和大小（共 12 个基准测试）上的实验表明，SHAPE 在 7B 模型中实现了显著收益，例如，在 MMVet（综合评估）中提高了 11.3%，在 MMBench（通用 VQA）中提高了 1.4%，在 POPE（幻觉稳健性）中提高了 8.0%。值得注意的是，定性分析证实了对视觉细节的更多关注以及整体描述与人类偏好的更好对齐。 

---
# One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs 

**Title (ZH)**: 一击即中：将多轮攻击 Consolidate 为高效单轮提示以应用于大语言模型 

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04856)  

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures. 

**Abstract (ZH)**: 尽管在大型语言模型（LLMs）中进行了广泛的安全增强，但由熟练的人类对手设计的多轮“监狱突破”对话仍然可以突破最复杂的防护措施。然而，这些多轮攻击需要大量的手动努力，限制了其可扩展性。在本工作中，我们介绍了一种名为多轮到单轮（M2S）的新方法，该方法系统地将多轮监狱突破提示转换为单轮攻击。具体而言，我们提出了三种转换策略——减号化、编号化和Python化，每种策略保留了顺序上下文，但将其打包成单一查询。我们在多轮人类监狱突破（MHJ）数据集上的实验表明，M2S通常能够提高或保持较高的攻击成功率（ASRs），与原始的多轮对话相比。值得注意的是，通过对有害性的StrongREJECT评估，M2S在Mistral-7B上的ASR达到了95.9%，而在GPT-4o上绝对提高了17.5%的性能。进一步的分析表明，某些对抗性策略，在合并为单一提示后，利用结构格式化提示来规避标准策略检查。这些发现表明，虽然单一轮次攻击比其多轮对手更为简单和经济，但它们仍然同样具有强大甚至更强大的威力。我们的发现强调了重新评估和加强LLM安全策略的紧迫性，特别是在对抗性查询可以压缩成单一提示但仍保留足够复杂性以规避现有安全措施的情况下。 

---
# Enhancing Collective Intelligence in Large Language Models Through Emotional Integration 

**Title (ZH)**: 通过情绪整合增强大型语言模型的集体智能 

**Authors**: Likith Kadiyala, Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2503.04849)  

**Abstract**: This research investigates the integration of emotional diversity into Large Language Models (LLMs) to enhance collective intelligence. Inspired by the human wisdom of crowds phenomenon, where group decisions often outperform individual judgments, we fine-tuned the DarkIdol-Llama-3.1-8B model using Google's GoEmotions dataset and Low-Rank Adaptation (LoRA) to simulate emotionally diverse responses. Evaluating the model on a distance estimation task between Fargo, ND, and Seattle, WA, across 15,064 unique persona configurations, we analyzed how emotional states and social attributes influence decision-making. Our findings demonstrate that emotional integration shapes response patterns while maintaining acceptable prediction accuracy, revealing its potential to enhance artificial collective intelligence. This study provides valuable insights into the interplay of emotional diversity and decision-making in LLMs, suggesting pathways for creating emotionally aware AI systems that balance emotional depth with analytical precision. 

**Abstract (ZH)**: 本研究调查了将情感多样性整合到大规模语言模型中以增强集体智能的方法。受人群中智慧群体现象的启发，该现象表明集体决策往往优于个人判断，我们使用Google的GoEmotions数据集和低秩适应（LoRA）对DarkIdol-Llama-3.1-8B模型进行了微调，以模拟情感多样化的响应。在对北达科他州法戈市和华盛顿州西雅图市之间距离估计任务的评估中，我们研究了15,064种独特的人设配置下情感状态和社会属性如何影响决策。研究结果表明，情感整合改变了响应模式，同时保持了可接受的预测准确性，揭示了其增强人工集体智能的潜力。本研究提供了关于情感多样性与大规模语言模型中决策相互作用的有价值见解，建议了创建既具备情感深度又兼具分析精确性的感知情感AI系统的途径。 

---
# Framing the Game: How Context Shapes LLM Decision-Making 

**Title (ZH)**: 框架游戏：背景如何塑造大语言模型决策-making 

**Authors**: Isaac Robinson, John Burden  

**Link**: [PDF](https://arxiv.org/pdf/2503.04840)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse contexts to support decision-making. While existing evaluations effectively probe latent model capabilities, they often overlook the impact of context framing on perceived rational decision-making. In this study, we introduce a novel evaluation framework that systematically varies evaluation instances across key features and procedurally generates vignettes to create highly varied scenarios. By analyzing decision-making patterns across different contexts with the same underlying game structure, we uncover significant contextual variability in LLM responses. Our findings demonstrate that this variability is largely predictable yet highly sensitive to framing effects. Our results underscore the need for dynamic, context-aware evaluation methodologies for real-world deployments. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不同情境中日益被部署以支持决策。虽然现有的评估有效探测了潜在模型能力，但往往忽视了情境框架对感知理性决策的影响。本研究引入了一种新的评估框架，系统地在关键特征上变化评估实例，并通过程序生成情景片断，创造高度多样的场景。通过在具有相同底层游戏结构的不同情境下分析决策模式，我们揭示了LLM响应中的显著情境变化性。我们的研究发现，这种变化性是高度可预测的，但对框架效应极为敏感。结果强调了在实际部署中需要动态的情境感知评估方法的重要性。 

---
# Extrapolation Merging: Keep Improving With Extrapolation and Merging 

**Title (ZH)**: 外推融合：通过外推和融合持续改进 

**Authors**: Yiguan Lin, Bin Xu, Yinghao Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04834)  

**Abstract**: Large Language Models (LLMs) require instruction fine-tuning to perform different downstream tasks. However, the instruction fine-tuning phase still demands significant computational resources and labeled data, lacking a paradigm that can improve model performance without additional computational power and data. Model merging aims to enhance performance by combining the parameters of different models, but the lack of a clear optimization direction during the merging process does not always guarantee improved performance. In this paper, we attempt to provide a clear optimization direction for model merging. We first validate the effectiveness of the model extrapolation method during the instruction fine-tuning phase. Then, we propose Extrapolation Merging, a paradigm that can continue improving model performance without requiring extra computational resources or data. Using the extrapolation method, we provide a clear direction for model merging, achieving local optimization search, and consequently enhancing the merged model's performance. We conduct experiments on seven different tasks, and the results show that our method can consistently improve the model's performance after fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）需要指令微调以执行不同的下游任务。然而，指令微调阶段仍然需要大量计算资源和标注数据，缺乏一种可以在不增加计算资源和数据的情况下提升模型性能的范式。模型合并旨在通过合并不同模型的参数来提高性能，但在合并过程中缺乏明确的优化方向并不总是能够保证性能的提升。本文尝试为模型合并提供一个明确的优化方向。我们首先验证了指令微调阶段模型外推方法的有效性。然后，我们提出了外推合并这一范式，可以在不需额外计算资源和数据的情况下继续提升模型性能。利用外推方法，我们为模型合并提供了一个明确的方向，实现了局部优化搜索，从而提升了合并模型的性能。我们在七个不同的任务上进行了实验，结果表明，我们的方法可以在微调后一致地提升模型的性能。 

---
# Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks 

**Title (ZH)**: 对抗训练以抵御 Jailbreak 攻击的多模态大型语言模型 

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.04833)  

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems. 

**Abstract (ZH)**: 基于对抗训练的多模态大型语言模型防范狱 break 攻击框架 

---
# "Only ChatGPT gets me": An Empirical Analysis of GPT versus other Large Language Models for Emotion Detection in Text 

**Title (ZH)**: “只有ChatGPT能打动我”：基于文本情绪检测的GPT与其他大型语言模型的实证分析 

**Authors**: Florian Lecourt, Madalina Croitoru, Konstantin Todorov  

**Link**: [PDF](https://arxiv.org/pdf/2503.04831)  

**Abstract**: This work investigates the capabilities of large language models (LLMs) in detecting and understanding human emotions through text. Drawing upon emotion models from psychology, we adopt an interdisciplinary perspective that integrates computational and affective sciences insights. The main goal is to assess how accurately they can identify emotions expressed in textual interactions and compare different models on this specific task. This research contributes to broader efforts to enhance human-computer interaction, making artificial intelligence technologies more responsive and sensitive to users' emotional nuances. By employing a methodology that involves comparisons with a state-of-the-art model on the GoEmotions dataset, we aim to gauge LLMs' effectiveness as a system for emotional analysis, paving the way for potential applications in various fields that require a nuanced understanding of human language. 

**Abstract (ZH)**: 本研究探讨了大型语言模型（LLMs）在通过文本检测和理解人类情绪方面的能力。借鉴心理学中的情绪模型，我们采取跨学科视角，结合计算科学和情感科学的见解。主要目标是评估它们在识别文本交互中表达的情绪方面的准确度，并将不同模型在此特定任务上进行对比。本研究为提升人机交互水平，使人工智能技术更加敏感和细腻地捕捉用户的情绪细微差别做出了贡献。通过在GoEmotions数据集上与最先进的模型进行比较的方法，我们旨在评估LLMs作为情绪分析系统的有效性，为需要细致理解人类语言的各种领域开辟潜在应用途径。 

---
# Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents 

**Title (ZH)**: 引用再发言：增强电子商务对话型LLM代理的情境-响应关联 

**Authors**: Jingying Zeng, Hui Liu, Zhenwei Dai, Xianfeng Tang, Chen Luo, Samarth Varshney, Zhen Li, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2503.04830)  

**Abstract**: With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers answer questions and smooth their shopping journey in e-commerce domain. The primary objective in building a trustworthy CSA is to ensure the agent's responses are accurate and factually grounded, which is essential for building customer trust and encouraging continuous engagement. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address these challenges, we present an easily productionized solution that enables a "citation experience" utilizing In-context Learning (ICL) and Multi-UX-Inference (MUI) to generate responses with citations to attribute its original sources without interfering other existing UX features. With proper UX design, these citation marks can be linked to the related product information and display the source to our customers. In this work, we also build auto-metrics and scalable benchmarks to holistically evaluate LLM's grounding and attribution capabilities. Our experiments demonstrate that incorporating this citation generation paradigm can substantially enhance the grounding of LLM responses by 13.83% on the real-world data. As such, our solution not only addresses the immediate challenges of LLM grounding issues but also adds transparency to conversational AI. 

**Abstract (ZH)**: 基于大规模语言模型的可信赖对话型购物代理的发展：利用上下文学习和多用户体验推断的引证经验实现语义接地与知识溯源 

---
# Beyond Next Word Prediction: Developing Comprehensive Evaluation Frameworks for measuring LLM performance on real world applications 

**Title (ZH)**: 超越下一个词预测：开发全面评价框架以衡量大模型在实际应用中的性能 

**Authors**: Vishakha Agrawal, Archie Chaudhury, Shreya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2503.04828)  

**Abstract**: While Large Language Models (LLMs) are fundamentally next-token prediction systems, their practical applications extend far beyond this basic function. From natural language processing and text generation to conversational assistants and software use, LLMs have numerous use-cases, and have already acquired a significant degree of enterprise adoption. To evaluate such models, static evaluation datasets, consisting of a set of prompts and their corresponding ground truths, are often used to benchmark the efficacy of the model for a particular task. In this paper, we provide the basis for a more comprehensive evaluation framework, based upon a traditional game and tool-based architecture that enables a more overarching measurement of a model's capabilities. For simplicity, we provide a generalized foundation that can be extended, without significant alteration, to numerous scenarios, from specific use cases such as supply chain management or financial reasoning, to abstract measurements such as ethics or safety. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）本质上是下一-token预测系统，但它们的实际应用远远超出了这一基本功能。从自然语言处理和文本生成到对话助手和软件使用，LLMs 有着众多的应用场景，并已获得了相当程度的企业采用。为了评估这些模型，通常会使用包含一组提示及其对应真实结果的静态评估数据集来衡量模型在特定任务上的有效性。在本文中，我们提供了一种更为全面的评价框架的基础，基于传统游戏和工具架构，能够更全面地衡量模型的能力。为了简化，我们提供了可以扩展且无需显著修改的基础框架，适用于从具体应用场景如供应链管理或金融推理，到抽象指标如伦理或安全性的多种情景。 

---
# Prompting Science Report 1: Prompt Engineering is Complicated and Contingent 

**Title (ZH)**: Prompt Engineering 是复杂且依赖环境的。 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2503.04818)  

**Abstract**: This is the first of a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we demonstrate two things:
- There is no single standard for measuring whether a Large Language Model (LLM) passes a benchmark, and that choosing a standard has a big impact on how well the LLM does on that benchmark. The standard you choose will depend on your goals for using an LLM in a particular case.
- It is hard to know in advance whether a particular prompting approach will help or harm the LLM's ability to answer any particular question. Specifically, we find that sometimes being polite to the LLM helps performance, and sometimes it lowers performance. We also find that constraining the AI's answers helps performance in some cases, though it may lower performance in other cases.
Taken together, this suggests that benchmarking AI performance is not one-size-fits-all, and also that particular prompting formulas or approaches, like being polite to the AI, are not universally valuable. 

**Abstract (ZH)**: 这是关于通过严谨测试帮助商业、教育和政策领导者理解与人工智能合作技术细节的一系列简短报告中的第一篇。本报告展示了两方面内容：
- 没有一种统一的标准来衡量大型语言模型（LLM）是否通过了一项基准测试，选择不同的标准会影响LLM在该基准测试中的表现。你选择的标准将取决于你在特定情况下使用LLM的目标。
- 在预先确定某一特定提示方法是否有助于或损害LLM回答特定问题的能力方面存在困难。具体而言，我们发现有时对LLM礼貌有助于提高性能，有时则会降低性能。我们还发现，限制AI的回答在某些情况下有助于提高性能，但在其他情况下则可能降低性能。
综上所述，这表明评估AI性能并非一刀切的，特定的提示公式或方法，如对AI礼貌，也不是普遍有价值的。 

---
# Self-Evolved Preference Optimization for Enhancing Mathematical Reasoning in Small Language Models 

**Title (ZH)**: 自我进化偏好优化以增强小型语言模型的数学推理能力 

**Authors**: Joykirat Singh, Tanmoy Chakraborty, Akshay Nambi  

**Link**: [PDF](https://arxiv.org/pdf/2503.04813)  

**Abstract**: Large language models (LLMs) have significantly improved their reasoning capabilities; however, they still struggle with complex multi-step mathematical problem-solving due to error propagation, lack of self-correction, and limited adaptability to diverse reasoning styles. Existing methods rely on static fine-tuning or prompt engineering, which fail to generalize across problem complexities, while the scarcity of high-quality preference data further hinders reliable reasoning.
We introduce SPHERE, a self-evolving data generation pipeline that enhances reasoning in small language models (SLMs) by iteratively generating, correcting, and diversifying reasoning chains. SPHERE operates in three stages: (i) Self-Generation, where the model autonomously constructs problem-solving steps; (ii) Self-Correction, enabling it to identify and rectify errors; and (iii) Diversity Induction, improving robustness through multiple valid reasoning trajectories. This self-evolution mechanism strengthens mathematical reasoning and enhances model reliability. Evaluations on MATH 500, GSM8K, AIME, AMC, and Olympiad show that SPHERE-trained models achieve significant gains over their base versions and match/surpass GPT-4o on certain benchmarks. Our findings demonstrate that self-evolving models can close the reasoning gap between SLMs and state-of-the-art LLMs, making mathematical AI more reliable, scalable, and efficient. 

**Abstract (ZH)**: 自适应数据生成pipeline增强小型语言模型的数学推理能力：SPHERE 

---
# LLaVE: Large Language and Vision Embedding Models with Hardness-Weighted Contrastive Learning 

**Title (ZH)**: LLaVE: 大规模语言和视觉嵌入模型与硬度加权对比学习 

**Authors**: Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.04812)  

**Abstract**: Universal multimodal embedding models play a critical role in tasks such as interleaved image-text retrieval, multimodal RAG, and multimodal clustering. However, our empirical results indicate that existing LMM-based embedding models trained with the standard InfoNCE loss exhibit a high degree of overlap in similarity distribution between positive and negative pairs, making it challenging to distinguish hard negative pairs effectively. To deal with this issue, we propose a simple yet effective framework that dynamically improves the embedding model's representation learning for negative pairs based on their discriminative difficulty. Within this framework, we train a series of models, named LLaVE, and evaluate them on the MMEB benchmark, which covers 4 meta-tasks and 36 datasets. Experimental results show that LLaVE establishes stronger baselines that achieve state-of-the-art (SOTA) performance while demonstrating strong scalability and efficiency. Specifically, LLaVE-2B surpasses the previous SOTA 7B models, while LLaVE-7B achieves a further performance improvement of 6.2 points. Although LLaVE is trained on image-text data, it can generalize to text-video retrieval tasks in a zero-shot manner and achieve strong performance, demonstrating its remarkable potential for transfer to other embedding tasks. 

**Abstract (ZH)**: 通用多模态嵌入模型在交错图像-文本检索、多模态RAG和多模态聚类等任务中发挥着关键作用。然而，我们的实验证据表明，使用标准InfoNCE损失训练的现有基于LLM的嵌入模型，正样本和负样本对之间的相似性分布存在高度重叠，这使得有效区分困难负样本对变得具有挑战性。为了解决这一问题，我们提出了一种简单而有效的框架，该框架根据负样本对的区分难度动态改进嵌入模型的表示学习。在该框架中，我们训练了一系列名为LLaVE的模型，并在MMEB基准上评估它们，该基准涵盖了4个元任务和36个数据集。实验结果表明，LLaVE建立了更强的基础模型，实现了最先进的性能，同时展示了强大的可扩展性和效率。特别地，LLaVE-2B超越了之前的7B模型，而LLaVE-7B进一步提高了6.2个百分点。尽管LLaVE是在图像-文本数据上训练的，但它可以在零样本情况下泛化到文本-视频检索任务，并实现强劲的性能，显示出它在其他嵌入任务上转移的巨大潜力。 

---
# PanguIR Technical Report for NTCIR-18 AEOLLM Task 

**Title (ZH)**: PanguIR 技术报告：NTCIR-18 AEOLLM 任务 

**Authors**: Lang Mei, Chong Chen, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04809)  

**Abstract**: As large language models (LLMs) gain widespread attention in both academia and industry, it becomes increasingly critical and challenging to effectively evaluate their capabilities. Existing evaluation methods can be broadly categorized into two types: manual evaluation and automatic evaluation. Manual evaluation, while comprehensive, is often costly and resource-intensive. Conversely, automatic evaluation offers greater scalability but is constrained by the limitations of its evaluation criteria (dominated by reference-based answers). To address these challenges, NTCIR-18 introduced the AEOLLM (Automatic Evaluation of LLMs) task, aiming to encourage reference-free evaluation methods that can overcome the limitations of existing approaches. In this paper, to enhance the evaluation performance of the AEOLLM task, we propose three key methods to improve the reference-free evaluation: 1) Multi-model Collaboration: Leveraging multiple LLMs to approximate human ratings across various subtasks; 2) Prompt Auto-optimization: Utilizing LLMs to iteratively refine the initial task prompts based on evaluation feedback from training samples; and 3) In-context Learning (ICL) Optimization: Based on the multi-task evaluation feedback, we train a specialized in-context example retrieval model, combined with a semantic relevance retrieval model, to jointly identify the most effective in-context learning examples. Experiments conducted on the final dataset demonstrate that our approach achieves superior performance on the AEOLLM task. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在学术界和工业界中获得广泛关注，有效地评价其能力变得日益重要和具有挑战性。现有的评价方法大致可以分为两类：人工评价和自动评价。人工评价虽然全面，但往往成本高昂且资源密集。相反，自动评价具有更高的可扩展性，但在评价标准的限制下（主要依赖参考答案）受到约束。为了应对这些挑战，NTCIR-18 引入了 AEOLLM（自动评价的LLMs）任务，旨在鼓励克服现有方法局限性的无参考评价方法。在本文中，为了提高AEOLLM任务的评价性能，我们提出了三种关键方法以改进无参考评价：1）多模型协作：利用多个LLM在各种子任务中近似人类评分；2）提示自动优化：使用LLM基于训练样本的评价反馈迭代优化初始任务提示；3）上下文相关学习（ICL）优化：基于多任务评价反馈，我们训练一个专门的上下文相关示例检索模型，结合语义相关性检索模型，共同识别最有效的上下文相关学习示例。在最终数据集上的实验表明，我们的方法在AEOLLM任务上取得了更好的性能。 

---
# Learning from Failures in Multi-Attempt Reinforcement Learning 

**Title (ZH)**: 基于多尝试强化学习中失败的学习 

**Authors**: Stephen Chung, Wenyu Du, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04808)  

**Abstract**: Recent advancements in reinforcement learning (RL) for large language models (LLMs), exemplified by DeepSeek R1, have shown that even a simple question-answering task can substantially improve an LLM's reasoning capabilities. In this work, we extend this approach by modifying the task into a multi-attempt setting. Instead of generating a single response per question, the model is given multiple attempts, with feedback provided after incorrect responses. The multi-attempt task encourages the model to refine its previous attempts and improve search efficiency. Experimental results show that even a small LLM trained on a multi-attempt task achieves significantly higher accuracy when evaluated with more attempts, improving from 45.6% with 1 attempt to 52.5% with 2 attempts on the math benchmark. In contrast, the same LLM trained on a standard single-turn task exhibits only a marginal improvement, increasing from 42.3% to 43.2% when given more attempts during evaluation. The results indicate that, compared to the standard single-turn task, an LLM trained on a multi-attempt task achieves slightly better performance on math benchmarks while also learning to refine its responses more effectively based on user feedback. Full code is available at this https URL 

**Abstract (ZH)**: Recent advancements in reinforcement learning for large language models: Extending reasoning capabilities through a multi-attempt task 

---
# Call for Rigor in Reporting Quality of Instruction Tuning Data 

**Title (ZH)**: 呼吁在报告指令调优数据质量方面保持严谨。 

**Authors**: Hyeonseok Moon, Jaehyung Seo, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04807)  

**Abstract**: Instruction tuning is crucial for adapting large language models (LLMs) to align with user intentions. Numerous studies emphasize the significance of the quality of instruction tuning (IT) data, revealing a strong correlation between IT data quality and the alignment performance of LLMs. In these studies, the quality of IT data is typically assessed by evaluating the performance of LLMs trained with that data. However, we identified a prevalent issue in such practice: hyperparameters for training models are often selected arbitrarily without adequate justification. We observed significant variations in hyperparameters applied across different studies, even when training the same model with the same data. In this study, we demonstrate the potential problems arising from this practice and emphasize the need for careful consideration in verifying data quality. Through our experiments on the quality of LIMA data and a selected set of 1,000 Alpaca data points, we demonstrate that arbitrary hyperparameter decisions can make any arbitrary conclusion. 

**Abstract (ZH)**: 指令调优对于使大型语言模型（LLMs）与用户意图相一致至关重要。众多研究强调了指令调优（IT）数据质量的 significance，指出IT数据质量与LLMs的对齐性能之间存在强烈的相关性。在这些研究中，通常通过评估使用该数据训练的LLMs的性能来评估IT数据质量。然而，我们发现这种做法中存在一个普遍问题：模型训练参数通常会被随意选择而缺乏充分的解释。我们观察到，在使用相同数据训练相同模型的情况下，不同研究中应用的参数存在显著差异。在本研究中，我们展示了这种做法可能导致的问题，并强调了验证数据质量时需要谨慎。通过在LIMA数据质量和1,000个Alpaca数据点上进行的实验，我们证明了任意选择超参数会导致任意结论。 

---
# Exploring and Evaluating Multimodal Knowledge Reasoning Consistency of Multimodal Large Language Models 

**Title (ZH)**: 探索和评估多模态大型语言模型的多模态知识推理一致性 

**Authors**: Boyu Jia, Junzhe Zhang, Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2503.04801)  

**Abstract**: In recent years, multimodal large language models (MLLMs) have achieved significant breakthroughs, enhancing understanding across text and vision. However, current MLLMs still face challenges in effectively integrating knowledge across these modalities during multimodal knowledge reasoning, leading to inconsistencies in reasoning outcomes. To systematically explore this issue, we propose four evaluation tasks and construct a new dataset. We conduct a series of experiments on this dataset to analyze and compare the extent of consistency degradation in multimodal knowledge reasoning within MLLMs. Based on the experimental results, we identify factors contributing to the observed degradation in consistency. Our research provides new insights into the challenges of multimodal knowledge reasoning and offers valuable guidance for future efforts aimed at improving MLLMs. 

**Abstract (ZH)**: 近年来，多模态大型语言模型（MLLMs）在文本和视觉理解方面取得了显著突破，但在多模态知识推理过程中仍面临有效整合知识的挑战，导致推理结果不一致。为系统探索这一问题，我们提出了四种评估任务并构建了一个新的数据集。在该数据集上进行了一系列实验，以分析和比较多模态知识推理中MLLMs一致性退化的程度。基于实验结果，我们识别出了导致一致性退化的因素。本研究为多模态知识推理面临的挑战提供了新的见解，并为未来改进MLLMs的努力提供了有价值的经验指导。 

---
# HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on Retrieval-Augmented Generation 

**Title (ZH)**: HoH：评估过时信息对检索增强生成影响的动态基准 

**Authors**: Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan, Yucong Luo, Jiaying Lin, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.04800)  

**Abstract**: While Retrieval-Augmented Generation (RAG) has emerged as an effective approach for addressing the knowledge outdating problem in Large Language Models (LLMs), it faces a critical challenge: the prevalence of outdated information in knowledge bases. Current research primarily focuses on incorporating up-to-date information, yet the impact of outdated information coexisting in retrieval sources remains inadequately addressed. To bridge this gap, we introduce HoH, the first benchmark specifically designed to evaluate the impact of outdated information on RAG. Our benchmark leverages token-level diff algorithms combined with LLM pipelines to efficiently create a large-scale QA dataset that accurately captures temporal knowledge evolution in real-world facts. Through comprehensive experiments, we reveal that outdated information significantly degrades RAG performance in two critical ways: (1) it substantially reduces response accuracy by distracting models from correct information, and (2) it can mislead models into generating potentially harmful outputs, even when current information is available. Current RAG approaches struggle with both retrieval and generation aspects when handling outdated information. These findings highlight the urgent need for innovative solutions to address the temporal challenges in RAG. 

**Abstract (ZH)**: HOH：评估检索增强生成中过时信息影响的第一个基准 

---
# Cyber for AI at SemEval-2025 Task 4: Forgotten but Not Lost: The Balancing Act of Selective Unlearning in Large Language Models 

**Title (ZH)**: Cyber for AI at SemEval-2025 Task 4: 忘记但未消失：大型语言模型中选择性遗忘的平衡艺术 

**Authors**: Dinesh Srivasthav P, Bala Mallikarjunarao Garlapati  

**Link**: [PDF](https://arxiv.org/pdf/2503.04795)  

**Abstract**: Large Language Models (LLMs) face significant challenges in maintaining privacy, ethics, and compliance, when sensitive or obsolete data must be selectively removed. Retraining these models from scratch is computationally infeasible, necessitating efficient alternatives. As part of the SemEval 2025 Task 4, this work focuses on the application of selective unlearning in LLMs to address this challenge. In this paper, we present our experiments and findings, primarily leveraging global weight modification to achieve an equilibrium between effectiveness of unlearning, knowledge retention, and target model's post-unlearning utility. We also detail the task-specific evaluation mechanism, results, and challenges. Our algorithms have achieved an aggregate score of 0.409 and 0.389 on the test set for 7B and 1B target models, respectively, demonstrating promising results in verifiable LLM unlearning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理敏感或过时数据的有选择性删除时面临着维护隐私、伦理和合规性的重大挑战。从头重新训练这些模型是计算上不可行的，因此需要高效的替代方案。作为SemEval 2025 Task 4的一部分，本工作聚焦于在LLMs中应用有选择性遗忘以应对这一挑战。在本文中，我们介绍了我们的实验和发现，主要通过全局权重修改来平衡遗忘效果、知识保留以及目标模型删除后的实用性。我们还详细描述了任务特定的评估机制、结果及挑战。我们的算法在7B和1B目标模型的测试集上分别获得了0.409和0.389的综合得分，展示了在可验证的LLM遗忘方面的有希望的结果。 

---
# Cross-linguistic disagreement as a conflict of semantic alignment norms in multilingual AI~Linguistic Diversity as a Problem for Philosophy, Cognitive Science, and AI~ 

**Title (ZH)**: 跨语言分歧作为多语言AI中语义对齐规范的冲突问题 Linguistic多样性对哲学、认知科学和AI的挑战 

**Authors**: Masaharu Mizumoto, Dat Tien Nguyen, Justin Sytsma, Mark Alfano, Yu Izumi, Koji Fujita, Nguyen Le Minh  

**Link**: [PDF](https://arxiv.org/pdf/2503.04792)  

**Abstract**: Multilingual large language models (LLMs) face an often-overlooked challenge stemming from intrinsic semantic differences across languages. Linguistic divergence can sometimes lead to cross-linguistic disagreements--disagreements purely due to semantic differences about a relevant concept. This paper identifies such disagreements as conflicts between two fundamental alignment norms in multilingual LLMs: cross-linguistic consistency (CL-consistency), which seeks universal concepts across languages, and consistency with folk judgments (Folk-consistency), which respects language-specific semantic norms. Through examining responses of conversational multilingual AIs in English and Japanese with the cases used in philosophy (cases of knowledge-how attributions), this study demonstrates that even state-of-the-art LLMs provide divergent and internally inconsistent responses. Such findings reveal a novel qualitative limitation in crosslingual knowledge transfer, or conceptual crosslingual knowledge barriers, challenging the assumption that universal representations and cross-linguistic transfer capabilities are inherently desirable. Moreover, they reveal conflicts of alignment policies of their developers, highlighting critical normative questions for LLM researchers and developers. The implications extend beyond technical alignment challenges, raising normative, moral-political, and metaphysical questions about the ideals underlying AI development--questions that are shared with philosophers and cognitive scientists but for which no one yet has definitive answers, inviting a multidisciplinary approach to balance the practical benefits of cross-linguistic consistency and respect for linguistic diversity. 

**Abstract (ZH)**: 多语言大型语言模型面临的内生语义差异引发的Often-Overlooked挑战：跨语言分歧与概念跨语言壁垒 

---
# Ext2Gen: Alignment through Unified Extraction and Generation for Robust Retrieval-Augmented Generation 

**Title (ZH)**: Ext2Gen：通过统一提取与生成进行稳健的检索增强生成对齐 

**Authors**: Hwanjun Song, Jeonghwan Choi, Minseok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04789)  

**Abstract**: Retrieval-augmented generation (RAG) enhances LLMs by integrating external knowledge, but generation remains fragile due to the uncertain placement of relevant chunks and retrieval-induced information overload, leading to hallucinations. We propose Ext2Gen, a novel extract-then-generate model that enhances RAG robustness by first extracting query-relevant sentences before generating answers. To optimize this model, we employ preference alignment through pairwise feedback learning, enabling the model to generate robust answers regardless of variations in retrieval results. Extensive experiments demonstrate that Ext2Gen effectively identifies query-relevant sentences with high precision and recall, leading to highly reliable answers. Furthermore, deploying our model in a RAG environment reveals that it not only boosts the performance of the base LLM but also synergizes with advanced retrieval strategies like query expansion. The dataset and model will be released soon. 

**Abstract (ZH)**: retrieve-然后生成增强（Ext2Gen）通过首先提取查询相关句子以增强RAG的稳定性，进而生成稳健的答案 

---
# AgroLLM: Connecting Farmers and Agricultural Practices through Large Language Models for Enhanced Knowledge Transfer and Practical Application 

**Title (ZH)**: 农林LLM：通过大型语言模型连接农民与农业实践以增强知识转移和实际应用 

**Authors**: Dinesh Jackson Samuel, Inna Skarga-Bandurova, David Sikolia, Muhammad Awais  

**Link**: [PDF](https://arxiv.org/pdf/2503.04788)  

**Abstract**: AgroLLM is an AI-powered chatbot designed to enhance knowledge-sharing and education in agriculture using Large Language Models (LLMs) and a Retrieval-Augmented Generation (RAG) framework. By using a comprehensive open-source agricultural database, AgroLLM provides accurate, contextually relevant responses while reducing incorrect information retrieval. The system utilizes the FAISS vector database for efficient similarity searches, ensuring rapid access to agricultural knowledge. A comparative study of three advanced models: Gemini 1.5 Flash, ChatGPT-4o Mini, and Mistral-7B-Instruct-v0.2 was conducted to evaluate performance across four key agricultural domains: Agriculture and Life Sciences, Agricultural Management, Agriculture and Forestry, and Agriculture Business. Key evaluation metrics included embedding quality, search efficiency, and response relevance. Results indicated that ChatGPT-4o Mini with RAG achieved the highest accuracy at 93%. Continuous feedback mechanisms enhance response quality, making AgroLLM a benchmark AI-driven educational tool for farmers, researchers, and professionals, promoting informed decision-making and improved agricultural practices. 

**Abstract (ZH)**: AgroLLM是一种利用大规模语言模型（LLMs）和检索增强生成（RAG）框架的AI驱动聊天机器人，旨在通过综合开源农业数据库增强农业领域的知识共享和教育。该系统利用FAISS向量数据库进行高效的相似搜索，确保快速访问农业知识。研究了三个先进模型：Gemini 1.5 Flash、ChatGPT-4o Mini和Mistral-7B-Instruct-v0.2在农业和生命科学、农业管理、农业和林业、农业商业四大关键农业领域中的性能。主要评估指标包括嵌入质量、搜索效率和响应相关性。结果显示，配备RAG的ChatGPT-4o Mini在准确性方面最高，达到93%。持续的反馈机制提高了响应质量，使AgroLLM成为农民、研究人员和专业人士的基准AI驱动教育工具，促进基于信息的决策和农业实践的改进。 

---
# Towards Anthropomorphic Conversational AI Part I: A Practical Framework 

**Title (ZH)**: Towards 类人对话人工智能 第一部分：一个实用框架 

**Authors**: Fei Wei, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.04787)  

**Abstract**: Large language models (LLMs), due to their advanced natural language capabilities, have seen significant success in applications where the user interface is usually a conversational artificial intelligence (AI) agent and engages the user through multi-round conversations. However, many scenarios require the agents to exhibit stronger social and conversational intelligence and demonstrate more human-like (anthropomorphic) reactions. This is an aspect that foundational LLMs have yet to fully address such that a single call of foundational models might be insufficient.
To bridge this gap, we propose a two-stage solution. In this work, we focus on the first stage, introducing a multi-module framework designed to replicate the key aspects of human intelligence involved in conversations. This framework comprises thinking modules for reasoning, resource modules for managing knowledge and external information, and response modules for generating contextually appropriate interactions. With all the modules cooperating, the framework would empower the agents to provide a better human-like conversation experience. In the second stage of our approach, these conversational data, after filtering and labeling, can serve as training and testing data for reinforcement learning, enabling AI to better capture human preferences. This stage is left for future work.
In our experiments, volunteers engaged in over 3000 rounds of conversation with the same AI character powered by a standalone LLM and our framework which integrates the same LLM. A separate group of evaluators rated the conversation samples, revealing that our framework significantly enhanced the social and conversational intelligence, even without fine-tuning the LLM. 

**Abstract (ZH)**: 大规模语言模型（LLMs）由于其先进的自然语言能力，在通常由会话人工智能（AI）代理器用户界面和通过多轮对话与用户交互的应用场景中取得了显著成功。然而，许多场景需要代理展现出更强的社会智能和交谈智慧，并表现出更多类似人类（拟人化）的反应。这是基础LLMs尚未充分解决的一个方面，因此基础模型单次调用可能不足以满足需求。

为此，我们提出了一种两阶段解决方案。在本文中，我们关注第一阶段，引入了一个多模块框架，旨在模仿对话中涉及的关键方面的人类智能。该框架包括推理模块、资源管理模块和响应模块，分别用于推理、管理和处理知识以及外部信息，以及生成上下文相关交互。通过所有模块的协同工作，该框架将赋予代理更好的拟人化对话体验。在我们方法的第二阶段，经过筛选和标注的对话数据可以作为强化学习的训练和测试数据，使AI更好地捕捉人类偏好。此阶段留待未来工作。

在我们的实验中，志愿者与由独立运行的LLM和我们结合在同一LLM的框架驱动的同一AI角色进行了超过3000轮对话。另外一组评估人员对对话样本进行了评估，结果显示我们的框架显著提高了社会智能和交谈智慧，即使没有对LLM进行微调。 

---
# KunlunBaize: LLM with Multi-Scale Convolution and Multi-Token Prediction Under TransformerX Framework 

**Title (ZH)**: KunlunBaize：在TransformerX框架下的多尺度卷积与多令牌预测大型语言模型 

**Authors**: Jiexiong Liu, Yixuan Chen, Yanqin Jia, Zhepeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.04784)  

**Abstract**: Large language models have demonstrated remarkable performance across various tasks, yet they face challenges such as low computational efficiency, gradient vanishing, and difficulties in capturing complex feature interactions. To address these limitations, a novel framework has been proposed. This framework incorporates a learnable dense residual skip connection mechanism, a TransformerX module a transformer based component integrating multiscale convolution and adaptive activation functions and a multitoken prediction interaction module. The learnable dense residual connections enhance information flow and feature capture across layers. Within the TransformerX module, large convolutional kernels aggregate semantic information from extensive text segments, while smaller convolutions focus on local word order and syntactic structures. The adaptive activation function dynamically adjusts its parameters based on the semantic features of the input text, improving the model's ability to handle diverse semantic expressions and complex relationships. The multitoken prediction module boosts data utilization and accelerates inference by predicting multiple future tokens. These components significantly enhance the performance and efficiency of large language models. 

**Abstract (ZH)**: 大型语言模型在各种任务中展现了出色的表现，但仍面临计算效率低、梯度消失以及捕捉复杂特征交互的难题。为解决这些问题，提出了一种新型框架。该框架包含可学习的密集_residual跳连机制、TransformerX模块（基于Transformer的部件，集成了多尺度卷积和自适应激活函数）以及多令牌预测交互模块。可学习的密集_residual跳连机制增强了各层间的信息流动和特征捕获。TransformerX模块通过大卷积核聚合大量文本段落的语义信息，而小卷积核则关注局部词序和句法结构。自适应激活函数根据输入文本的语义特征动态调整参数，提高了模型处理多种语义表达和复杂关系的能力。多令牌预测模块通过预测多个未来令牌来增强数据利用和加速推理。这些组件显著提升了大型语言模型的性能和效率。 

---
# MV-CLAM: Multi-View Molecular Interpretation with Cross-Modal Projection via Language Model 

**Title (ZH)**: MV-CLAM：通过语言模型实现跨模态投影的多视图分子解释 

**Authors**: Sumin Ha, Jun Hyeong Kim, Yinhua Piao, Sun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.04780)  

**Abstract**: Human expertise in chemistry and biomedicine relies on contextual molecular understanding, a capability that large language models (LLMs) can extend through fine-grained alignment between molecular structures and text. Recent multimodal learning advances focus on cross-modal alignment, but existing molecule-text models ignore complementary information in different molecular views and rely on single-view representations, limiting molecular understanding. Moreover, naïve multi-view alignment strategies face two challenges: (1) separate aligned spaces with inconsistent mappings between molecule and text embeddings, and that (2) existing loss objectives fail to preserve complementary information for fine-grained alignment. This can limit the LLM's ability to fully understand the molecular properties. To address these issues, we propose MV-CLAM, a novel framework that aligns multi-view molecular representations into a unified textual space using a multi-query transformer (MQ-Former). Our approach ensures cross-view consistency while a token-level contrastive loss preserves diverse molecular features across textual queries. MV-CLAM enhances molecular reasoning, improving retrieval and captioning accuracy. The source code of MV-CLAM is available in this https URL. 

**Abstract (ZH)**: 多视图CLAM：一种将多视图分子表示统一到文本空间的新型框架 

---
# Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference 

**Title (ZH)**: LLMs在形式化规范推断方面的推理能力探究：一项全面评估 

**Authors**: Thanh Le-Cong, Bach Le, Toby Murray  

**Link**: [PDF](https://arxiv.org/pdf/2503.04779)  

**Abstract**: Large Language Models (LLMs) are increasingly being used to automate programming tasks. Yet, LLMs' capabilities in reasoning about program semantics are still inadequately studied, leaving significant potential for further exploration. This paper introduces FormalBench, a comprehensive benchmark designed to evaluate LLMs' reasoning abilities on program semantics, particularly via the task of synthesizing formal program specifications to assist verifying program correctness. This task requires both comprehensive reasoning over all possible program executions (i.e., \textit{completeness}) and the generation of precise, syntactically correct expressions that adhere to formal syntax and semantics (i.e., \textit{consistency}). Using this benchmark, we evaluated the ability of LLMs in synthesizing consistent and complete specifications. Our findings show that LLMs perform well with simple control flows but struggle with more complex structures, especially loops, even with advanced prompting. Additionally, LLMs exhibit limited robustness against semantic-preserving transformations. We also highlight common failure patterns and design self-repair prompts, improving success rates by 25%. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用于自动化编程任务。然而，LLMs在程序语义推理方面的能力仍然研究不足，留下了进一步探索的巨大潜力。本文介绍了FormalBench，这是一个全面的基准，旨在评估LLMs在程序语义推理方面的能力，特别是通过合成正式的程序规范以辅助验证程序正确性的任务。这一任务要求对所有可能的程序执行进行全面推理（即完备性）和生成精确且符合正式语法和语义的表达式（即一致性）。利用这一基准，我们评估了LLMs在合成一致且完整的规范方面的能力。我们的研究结果表明，LLMs在简单的控制流方面表现良好，但在更复杂的结构，尤其是循环结构方面表现较差，即使使用高级提示也是如此。此外，LLMs对语义保留转换的鲁棒性有限。我们还指出了常见的失败模式，并设计了自我修复提示，将成功率提高了25%。 

---
# Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs 

**Title (ZH)**: 通过探索状态转换图生成数百万条精简定理及其证明 

**Authors**: David Yin, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04772)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant potential in generating mathematical proofs. However, a persistent challenge is that LLMs occasionally make mistakes, while even a minor mistake can invalidate an entire proof. Proof assistants like Lean offer a great remedy. They are designed for verifying each step of a proof in a formal language, and in recent years researchers have created AI models to generate proofs in their languages. However, the scarcity of large-scale datasets of Lean proofs restrict the performance of such Automated Theorem Proving (ATP) models.
We developed LeanNavigator, a novel method for generating a large-scale dataset of Lean theorems and proofs by finding new ways to prove existing Lean theorems. By leveraging an interactive Lean client and an efficient method for proof step generation, LeanNavigator efficiently produces new theorems with corresponding proofs. Applying this approach to Mathlib4, we generated 4.7 million theorems totaling 1 billion tokens, surpassing previous datasets by more than an order of magnitude. Using this extensive dataset, we trained an AI model that outperforms the state-of-the-art ReProver model in theorem-proving tasks. These results confirm our hypothesis and demonstrate the critical role of large datasets in improving the performance of automated theorem provers. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成数学证明方面展现了显著的潜力。然而，一个持续性的挑战是LLMs偶尔会出现错误，即使是一个小错误也可能使整个证明无效。像Lean这样的证明助手提供了很好的解决办法。它们旨在以形式语言验证每一个证明步骤，并且近年来研究人员已经创建了能够生成它们语言中证明的AI模型。然而，Lean证明的大规模数据集稀缺限制了此类自动化定理证明（ATP）模型的表现。我们开发了LeanNavigator，这是一种新颖的方法，通过寻找证明现有Lean定理的新途径来生成大规模的Lean定理和证明数据集。通过利用交互式Lean客户端和高效的证明步骤生成方法，LeanNavigator高效地产生了新的定理及其对应的证明。将这种方法应用于Mathlib4，我们生成了总计1亿个词元的470万个定理，大大超过了之前的数据集的规模。利用这个庞大的数据集，我们训练了一个AI模型，其在定理证明任务上超过了最先进的ReProver模型。这些结果证实了我们的假设，并展示了大规模数据集在提高自动化定理证明性能中的关键作用。 

---
# Which Economic Tasks are Performed with AI? Evidence from Millions of Claude Conversations 

**Title (ZH)**: 哪些经济任务是由AI执行的？来自数百万次Claude对话的证据 

**Authors**: Kunal Handa, Alex Tamkin, Miles McCain, Saffron Huang, Esin Durmus, Sarah Heck, Jared Mueller, Jerry Hong, Stuart Ritchie, Tim Belonax, Kevin K. Troy, Dario Amodei, Jared Kaplan, Jack Clark, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2503.04761)  

**Abstract**: Despite widespread speculation about artificial intelligence's impact on the future of work, we lack systematic empirical evidence about how these systems are actually being used for different tasks. Here, we present a novel framework for measuring AI usage patterns across the economy. We leverage a recent privacy-preserving system to analyze over four million this http URL conversations through the lens of tasks and occupations in the U.S. Department of Labor's O*NET Database. Our analysis reveals that AI usage primarily concentrates in software development and writing tasks, which together account for nearly half of all total usage. However, usage of AI extends more broadly across the economy, with approximately 36% of occupations using AI for at least a quarter of their associated tasks. We also analyze how AI is being used for tasks, finding 57% of usage suggests augmentation of human capabilities (e.g., learning or iterating on an output) while 43% suggests automation (e.g., fulfilling a request with minimal human involvement). While our data and methods face important limitations and only paint a picture of AI usage on a single platform, they provide an automated, granular approach for tracking AI's evolving role in the economy and identifying leading indicators of future impact as these technologies continue to advance. 

**Abstract (ZH)**: 尽管人们对人工智能对未来工作的影响进行了广泛猜测，但我们缺乏系统性实证证据来说明这些系统在不同任务中的实际使用模式。在此，我们提出了一种衡量经济领域中人工智能使用模式的新框架。我们利用一个近期的隐私保护系统，分析了超过400万条this http URL对话，从美国劳工部的O*NET数据库中的任务和职业视角进行分析。我们的分析揭示，人工智能的使用主要集中在软件开发和写作任务上，这两大类任务占据了总体使用量的近一半。然而，人工智能的使用更广泛地延伸到整个经济领域，大约有36%的职业在其相关任务中至少使用了四分之一的人工智能。我们还分析了人工智能在任务中的使用情况，发现57%的使用表明了对人类能力的增强（例如学习或改进输出），而43%的使用表明了自动化（例如在最少人类干预的情况下完成请求）。尽管我们的数据和方法存在重要限制，只能描绘单个平台中人工智能使用状况的图景，但它们提供了自动化的、详细的追踪人工智能在经济中 evolving 角色的方法，并有助于识别这些技术继续发展时未来影响的领先指标。 

---
# Peeking Behind Closed Doors: Risks of LLM Evaluation by Private Data Curators 

**Title (ZH)**: 窥视密室之内：基于私人数据策展人评估的大语言模型风险 

**Authors**: Hritik Bansal, Pratyush Maini  

**Link**: [PDF](https://arxiv.org/pdf/2503.04756)  

**Abstract**: The rapid advancement in building large language models (LLMs) has intensified competition among big-tech companies and AI startups. In this regard, model evaluations are critical for product and investment-related decision-making. While open evaluation sets like MMLU initially drove progress, concerns around data contamination and data bias have constantly questioned their reliability. As a result, it has led to the rise of private data curators who have begun conducting hidden evaluations with high-quality self-curated test prompts and their own expert annotators. In this paper, we argue that despite potential advantages in addressing contamination issues, private evaluations introduce inadvertent financial and evaluation risks. In particular, the key concerns include the potential conflict of interest arising from private data curators' business relationships with their clients (leading LLM firms). In addition, we highlight that the subjective preferences of private expert annotators will lead to inherent evaluation bias towards the models trained with the private curators' data. Overall, this paper lays the foundation for studying the risks of private evaluations that can lead to wide-ranging community discussions and policy changes. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的快速进展加剧了大-tech公司和AI初创企业的竞争。在这种背景下，模型评估对于产品和投资决策至关重要。虽然像MMLU这样的开放评估集最初促进了进展，但关于数据污染和数据偏见的担忧不断质疑其可靠性。因此，这导致了私有数据策划者的兴起，他们开始进行隐藏评估，并使用高质量的自策划测试提示和自己的专家注释者。在本文中，我们argue尽管私人评估可能在解决污染问题方面具有潜在优势，但它们引入了潜在的财务和评估风险。特别是，主要关注点包括私营数据策划者与其客户（领先LLM公司）的商业关系可能引发的利益冲突。此外，我们强调私营专家注释者的主观偏好将导致模型对私营策划者数据进行训练时固有的评估偏见。总体而言，本文为研究可能导致广泛社区讨论和政策变化的私人评估风险奠定了基础。 

---
# What can large language models do for sustainable food? 

**Title (ZH)**: 大型语言模型如何促进可持续食品？ 

**Authors**: Anna T. Thomas, Adam Yee, Andrew Mayne, Maya B. Mathur, Dan Jurafsky, Kristina Gligorić  

**Link**: [PDF](https://arxiv.org/pdf/2503.04734)  

**Abstract**: Food systems are responsible for a third of human-caused greenhouse gas emissions. We investigate what Large Language Models (LLMs) can contribute to reducing the environmental impacts of food production. We define a typology of design and prediction tasks based on the sustainable food literature and collaboration with domain experts, and evaluate six LLMs on four tasks in our typology. For example, for a sustainable protein design task, food science experts estimated that collaboration with an LLM can reduce time spent by 45% on average, compared to 22% for collaboration with another expert human food scientist. However, for a sustainable menu design task, LLMs produce suboptimal solutions when instructed to consider both human satisfaction and climate impacts. We propose a general framework for integrating LLMs with combinatorial optimization to improve reasoning capabilities. Our approach decreases emissions of food choices by 79% in a hypothetical restaurant while maintaining participants' satisfaction with their set of choices. Our results demonstrate LLMs' potential, supported by optimization techniques, to accelerate sustainable food development and adoption. 

**Abstract (ZH)**: 食物系统 responsable for 约三分之一的人为温室气体排放。我们探讨了大型语言模型（LLMs）在减少食物生产环境影响方面的贡献。我们根据可持续食物文献和与领域专家的合作，定义了一种设计和预测任务类型，并在我们定义的四种任务上评估了六种LLMs。例如，在可持续蛋白质设计任务中，食品科学专家估计与LLM合作可以将平均时间减少45%，而与另一名专家人类食品科学家合作则减少22%。然而，在可持续菜单设计任务中，当LLM被指示同时考虑人类满意和气候影响时，生成的解决方案往往是次优的。我们提出了一种通用框架，通过结合组合优化来增强LLM的推理能力。我们的方法在假设的餐馆中减少了79%的食物选择排放，同时保持了参与者对其选择集的满意度。我们的结果证明了，通过优化技术，LLMs有能力加速可持续食物的发展和采用。 

---
# Leveraging Large Language Models For Optimized Item Categorization using UNSPSC Taxonomy 

**Title (ZH)**: 利用大型语言模型优化基于UNSPSC分类法的物品分类 

**Authors**: Anmolika Singh, Yuhang Diao  

**Link**: [PDF](https://arxiv.org/pdf/2503.04728)  

**Abstract**: Effective item categorization is vital for businesses, enabling the transformation of unstructured datasets into organized categories that streamline inventory management. Despite its importance, item categorization remains highly subjective and lacks a uniform standard across industries and businesses. The United Nations Standard Products and Services Code (UNSPSC) provides a standardized system for cataloguing inventory, yet employing UNSPSC categorizations often demands significant manual effort. This paper investigates the deployment of Large Language Models (LLMs) to automate the classification of inventory data into UNSPSC codes based on Item Descriptions. We evaluate the accuracy and efficiency of LLMs in categorizing diverse datasets, exploring their language processing capabilities and their potential as a tool for standardizing inventory classification. Our findings reveal that LLMs can substantially diminish the manual labor involved in item categorization while maintaining high accuracy, offering a scalable solution for businesses striving to enhance their inventory management practices. 

**Abstract (ZH)**: 有效的物品分类对于企业至关重要，能够将无结构的数据集转化为有组织的类别，简化库存管理。尽管如此，物品分类仍然高度主观，并且缺乏跨行业和企业的统一标准。联合国标准产品和服务分类码（UNSPSC）提供了一种标准化的库存分类系统，但采用UNSPSC分类通常需要大量的手动努力。本文研究了大型语言模型（LLMs）在基于物品描述将库存数据分类到UNSPSC代码中的应用效果，评估了LLMs在分类多样数据集方面的准确性和效率，探讨了它们的语言处理能力和作为标准化库存分类工具的潜力。我们的研究发现，LLMs可以大幅减少物品分类所需的 manual 努力，同时保持高准确性，为企业提供了一种可扩展的解决方案，以增强其库存管理实践。 

---
