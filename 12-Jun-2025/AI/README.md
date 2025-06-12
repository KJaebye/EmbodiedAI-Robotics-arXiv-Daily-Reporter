# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning 

**Title (ZH)**: V-JEPA 2: 自监督视频模型实现理解、预测和规划 

**Authors**: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas  

**Link**: [PDF](https://arxiv.org/pdf/2506.09985)  

**Abstract**: A major challenge for modern AI is to learn to understand the world and learn to act largely by observation. This paper explores a self-supervised approach that combines internet-scale video data with a small amount of interaction data (robot trajectories), to develop models capable of understanding, predicting, and planning in the physical world. We first pre-train an action-free joint-embedding-predictive architecture, V-JEPA 2, on a video and image dataset comprising over 1 million hours of internet video. V-JEPA 2 achieves strong performance on motion understanding (77.3 top-1 accuracy on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100) surpassing previous task-specific models. Additionally, after aligning V-JEPA 2 with a large language model, we demonstrate state-of-the-art performance on multiple video question-answering tasks at the 8 billion parameter scale (e.g., 84.0 on PerceptionTest, 76.9 on TempCompass). Finally, we show how self-supervised learning can be applied to robotic planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on Franka arms in two different labs and enable picking and placing of objects using planning with image goals. Notably, this is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward. This work demonstrates how self-supervised learning from web-scale data and a small amount of robot interaction data can yield a world model capable of planning in the physical world. 

**Abstract (ZH)**: 现代AI面临的一个主要挑战是通过观察学习理解和行动。本文探讨了一种自监督方法，该方法结合了互联网规模的视频数据和少量交互数据（机器人轨迹），以开发能够在物理世界中理解、预测和规划的模型。首先，我们在包含超过100万小时互联网视频的视频和图像数据集上预训练了一个无动作联合嵌入预测架构V-JEPA 2。V-JEPA 2在动作理解方面表现出色（在Something-Something v2中取得77.3%的top-1准确性），在人类动作预见方面也取得了最先进的性能（在Epic-Kitchens-100中召回率达到39.7%），超越了之前的专业任务模型。此外，在将V-JEPA 2与大规模语言模型对齐后，我们展示了在80亿参数的大规模下多项视频问答任务的领先性能（例如，在PerceptionTest上取得84.0%，在TempCompass上取得76.9%）。最后，通过使用不到62小时的未标注机器人视频数据，我们展示了自监督学习在机器人规划任务中的应用，通过泛化训练出一个潜在动作条件的世界模型V-JEPA 2-AC，并部署在两个不同的实验室中的Franka手臂上，实现图像目标的物体抓取和放置。值得注意的是，这在这些环境内没有收集任何机器人数据，并且没有进行任何任务特定训练或奖励。本文展示了从网页规模数据和少量机器人交互数据进行自监督学习，能够生成能够在物理世界中规划的环境模型。 

---
# How Do People Revise Inconsistent Beliefs? Examining Belief Revision in Humans with User Studies 

**Title (ZH)**: 人们如何修订不一致的信念？通过用户研究考察人类信念修订過程 

**Authors**: Stylianos Loukas Vasileiou, Antonio Rago, Maria Vanina Martinez, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2506.09977)  

**Abstract**: Understanding how humans revise their beliefs in light of new information is crucial for developing AI systems which can effectively model, and thus align with, human reasoning. While theoretical belief revision frameworks rely on a set of principles that establish how these operations are performed, empirical evidence from cognitive psychology suggests that people may follow different patterns when presented with conflicting information. In this paper, we present three comprehensive user studies showing that people consistently prefer explanation-based revisions, i.e., those which are guided by explanations, that result in changes to their belief systems that are not necessarily captured by classical belief change theory. Our experiments systematically investigate how people revise their beliefs with explanations for inconsistencies, whether they are provided with them or left to formulate them themselves, demonstrating a robust preference for what may seem non-minimal revisions across different types of scenarios. These findings have implications for AI systems designed to model human reasoning or interact with humans, suggesting that such systems should accommodate explanation-based, potentially non-minimal belief revision operators to better align with human cognitive processes. 

**Abstract (ZH)**: 理解人类在新信息面前如何修订信念对于开发能够有效模拟和对齐人类推理的AI系统至关重要。虽然理论性的信念修订框架基于一套原则来确定这些操作的执行方式，但认知心理学的实证证据表明，当人们面临冲突信息时，可能会遵循不同的模式。在本文中，我们展示了三项全面的用户研究，表明人们一致偏好基于解释的修订，即那些受到解释指引、可能导致信念系统发生变化但不一定能被经典信念变更理论捕获的修订。我们的实验系统地探究了人们在不一致情况下如何根据解释修订信念，无论是由提供还是自行构想这些解释，展现了在不同场景类型中对似乎非最小化修订的稳健偏好。这些发现对旨在模拟人类推理或与人类交互的AI系统具有启示意义，建议此类系统应容纳基于解释的、可能非最小化的信念修订操作，以更好地与人类认知过程对齐。 

---
# Intent Factored Generation: Unleashing the Diversity in Your Language Model 

**Title (ZH)**: 意图因子生成：激发您语言模型中的多样性 

**Authors**: Eltayeb Ahmed, Uljad Berdica, Martha Elliott, Danijela Horak, Jakob N. Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.09659)  

**Abstract**: Obtaining multiple meaningfully diverse, high quality samples from Large Language Models for a fixed prompt remains an open challenge. Current methods for increasing diversity often only operate at the token-level, paraphrasing the same response. This is problematic because it leads to poor exploration on reasoning problems and to unengaging, repetitive conversational agents. To address this we propose Intent Factored Generation (IFG), factorising the sampling process into two stages. First, we sample a semantically dense intent, e.g., a summary or keywords. Second, we sample the final response conditioning on both the original prompt and the intent from the first stage. This allows us to use a higher temperature during the intent step to promote conceptual diversity, and a lower temperature during the final generation to ensure the outputs are coherent and self-consistent. Additionally, we find that prompting the model to explicitly state its intent for each step of the chain-of-thought before generating the step is beneficial for reasoning tasks. We demonstrate our method's effectiveness across a diverse set of tasks. We show this method improves both pass@k and Reinforcement Learning from Verifier Feedback on maths and code tasks. For instruction-tuning, we combine IFG with Direct Preference Optimisation to increase conversational diversity without sacrificing reward. Finally, we achieve higher diversity while maintaining the quality of generations on a general language modelling task, using a new dataset of reader comments and news articles that we collect and open-source. In summary, we present a simple method of increasing the sample diversity of LLMs while maintaining performance. This method can be implemented by changing the prompt and varying the temperature during generation, making it easy to integrate into many algorithms for gains across various applications. 

**Abstract (ZH)**: 从大型语言模型中为固定提示获得多种意义多样且高质量的样本仍然是一个开放性的挑战。当前增加多样性的方法往往仅在token级别进行操作，对响应进行改写。这在分析问题时会导致探索不足，并导致不吸引人、重复的对话代理。为了解决这一问题，我们提出了意图因子生成（IFG）方法，将采样过程分为两个阶段。首先，我们采样一个语义密集的意图，例如总结或关键词。然后，在第一阶段的基础上，我们在有条件生成最终响应时采样。这允许我们在意图步骤中使用更高的温度来促进概念多样性，并在最终生成时使用较低的温度以确保输出的连贯性和一致性。此外，我们发现，在生成每一步骤之前，明确提示模型陈述其意图是有益的，特别是在推理任务中。我们展示了该方法在多种任务上的有效性。我们证明了这种方法在数学和代码任务上提高了pass@k和验证器反馈强化学习的性能。对于指令调优，我们结合了IFG和直接偏好优化，以提高对话多样性而不牺牲奖励。最后，我们在一项通用语言建模任务上实现了更高的多样性，同时保持生成的质量，使用了一个新收集并开源的读者评论和新闻文章数据集。总之，我们提出了一种简单的方法，可以在保持性能的同时增加LLM的样本多样性。这种方法可以通过更改提示和在生成过程中调整温度来实现，使其易于集成到许多算法中，从而在各种应用中获得收益。 

---
# Application-Driven Value Alignment in Agentic AI Systems: Survey and Perspectives 

**Title (ZH)**: 代理AI系统中基于应用的价值对齐：综述与展望 

**Authors**: Wei Zeng, Hengshu Zhu, Chuan Qin, Han Wu, Yihang Cheng, Sirui Zhang, Xiaowei Jin, Yinuo Shen, Zhenxing Wang, Feimin Zhong, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.09656)  

**Abstract**: The ongoing evolution of AI paradigms has propelled AI research into the Agentic AI stage. Consequently, the focus of research has shifted from single agents and simple applications towards multi-agent autonomous decision-making and task collaboration in complex environments. As Large Language Models (LLMs) advance, their applications become more diverse and complex, leading to increasingly situational and systemic risks. This has brought significant attention to value alignment for AI agents, which aims to ensure that an agent's goals, preferences, and behaviors align with human values and societal norms. This paper reviews value alignment in agent systems within specific application scenarios. It integrates the advancements in AI driven by large models with the demands of social governance. Our review covers value principles, agent system application scenarios, and agent value alignment evaluation. Specifically, value principles are organized hierarchically from a top-down perspective, encompassing macro, meso, and micro levels. Agent system application scenarios are categorized and reviewed from a general-to-specific viewpoint. Agent value alignment evaluation systematically examines datasets for value alignment assessment and relevant value alignment methods. Additionally, we delve into value coordination among multiple agents within agent systems. Finally, we propose several potential research directions in this field. 

**Abstract (ZH)**: 人工智能范式的持续进化将AI研究推进到了自主智能Agent阶段。随之，研究重点从单一Agent和简单应用转向了复杂环境中的多Agent自主决策与任务协作。随着大型语言模型（LLMs）的发展，其应用变得越来越多样化和复杂，导致了越来越具体和系统的风险。这引起了对Agent的价值对齐的广泛关注，目的在于确保Agent的目标、偏好和行为与人类价值观和社会规范相一致。本文在特定应用场景下回顾Agent系统中的价值对齐。我们将由大型模型驱动的AI进步与社会治理的需求相结合。本回顾涵盖了价值原则、Agent系统应用场景以及Agent价值对齐评估。具体而言，从自上而下的层级视角组织价值原则，涵盖宏观、中观和微观层面；从一般到具体的视角分类并回顾Agent系统应用场景；系统性地检查数据集的价值对齐评估以及相关的价值对齐方法。此外，我们还探讨了Agent系统内部多个Agent之间的价值协调。最后，我们提出了该领域中若干潜在的研究方向。 

---
# DipLLM: Fine-Tuning LLM for Strategic Decision-making in Diplomacy 

**Title (ZH)**: DipLLM：细调大型语言模型以支持外交战略决策 

**Authors**: Kaixuan Xu, Jiajun Chai, Sicheng Li, Yuqian Fu, Yuanheng Zhu, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09655)  

**Abstract**: Diplomacy is a complex multiplayer game that requires both cooperation and competition, posing significant challenges for AI systems. Traditional methods rely on equilibrium search to generate extensive game data for training, which demands substantial computational resources. Large Language Models (LLMs) offer a promising alternative, leveraging pre-trained knowledge to achieve strong performance with relatively small-scale fine-tuning. However, applying LLMs to Diplomacy remains challenging due to the exponential growth of possible action combinations and the intricate strategic interactions among players. To address this challenge, we propose DipLLM, a fine-tuned LLM-based agent that learns equilibrium policies for Diplomacy. DipLLM employs an autoregressive factorization framework to simplify the complex task of multi-unit action assignment into a sequence of unit-level decisions. By defining an equilibrium policy within this framework as the learning objective, we fine-tune the model using only 1.5% of the data required by the state-of-the-art Cicero model, surpassing its performance. Our results demonstrate the potential of fine-tuned LLMs for tackling complex strategic decision-making in multiplayer games. 

**Abstract (ZH)**: 外交是一款既需合作又需竞争的复杂多人游戏，对AI系统构成了重大挑战。传统方法依赖于平衡搜索来生成大量游戏数据用于训练，这需要大量的计算资源。大规模语言模型（LLMs）提供了一种有前景的替代方案，利用预训练知识实现较强性能，仅需相对较小规模的微调。然而，将LLMs应用于外交仍面临挑战，因为可能的动作组合呈指数增长，且玩家之间的战略互动非常复杂。为应对这一挑战，我们提出DipLLM，这是一种基于微调语言模型的代理程序，学习外交的游戏平衡策略。DipLLM采用自回归分解框架，将复杂的多单位动作分配任务简化为一系列单元级别的决策。通过在此框架下定义平衡策略作为学习目标，我们仅使用先进Cicero模型所需数据的1.5%，即实现了超越其性能的结果。我们的结果展示了微调语言模型在处理多人游戏复杂战略决策方面的潜力。 

---
# Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning 

**Title (ZH)**: 快速蒙特卡洛树扩散：通过并行稀疏规划实现100倍加速 

**Authors**: Jaesik Yoon, Hyeonseo Cho, Yoshua Bengio, Sungjin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2506.09498)  

**Abstract**: Diffusion models have recently emerged as a powerful approach for trajectory planning. However, their inherently non-sequential nature limits their effectiveness in long-horizon reasoning tasks at test time. The recently proposed Monte Carlo Tree Diffusion (MCTD) offers a promising solution by combining diffusion with tree-based search, achieving state-of-the-art performance on complex planning problems. Despite its strengths, our analysis shows that MCTD incurs substantial computational overhead due to the sequential nature of tree search and the cost of iterative denoising. To address this, we propose Fast-MCTD, a more efficient variant that preserves the strengths of MCTD while significantly improving its speed and scalability. Fast-MCTD integrates two techniques: Parallel MCTD, which enables parallel rollouts via delayed tree updates and redundancy-aware selection; and Sparse MCTD, which reduces rollout length through trajectory coarsening. Experiments show that Fast-MCTD achieves up to 100x speedup over standard MCTD while maintaining or improving planning performance. Remarkably, it even outperforms Diffuser in inference speed on some tasks, despite Diffuser requiring no search and yielding weaker solutions. These results position Fast-MCTD as a practical and scalable solution for diffusion-based inference-time reasoning. 

**Abstract (ZH)**: 基于扩散模型的快速蒙特卡洛树搜索（Fast-MCTD）：一种高效的轨迹规划方法 

---
# A Call for Collaborative Intelligence: Why Human-Agent Systems Should Precede AI Autonomy 

**Title (ZH)**: 呼吁协作智能：为何人类代理系统应先于人工智能自主性发展 

**Authors**: Henry Peng Zou, Wei-Chieh Huang, Yaozu Wu, Chunyu Miao, Dongyuan Li, Aiwei Liu, Yue Zhou, Yankai Chen, Weizhi Zhang, Yangning Li, Liancheng Fang, Renhe Jiang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09420)  

**Abstract**: Recent improvements in large language models (LLMs) have led many researchers to focus on building fully autonomous AI agents. This position paper questions whether this approach is the right path forward, as these autonomous systems still have problems with reliability, transparency, and understanding the actual requirements of human. We suggest a different approach: LLM-based Human-Agent Systems (LLM-HAS), where AI works with humans rather than replacing them. By keeping human involved to provide guidance, answer questions, and maintain control, these systems can be more trustworthy and adaptable. Looking at examples from healthcare, finance, and software development, we show how human-AI teamwork can handle complex tasks better than AI working alone. We also discuss the challenges of building these collaborative systems and offer practical solutions. This paper argues that progress in AI should not be measured by how independent systems become, but by how well they can work with humans. The most promising future for AI is not in systems that take over human roles, but in those that enhance human capabilities through meaningful partnership. 

**Abstract (ZH)**: 近期大语言模型的进步促使许多研究者致力于构建完全自主的AI代理。这类文章质疑这种途径是否是前进的正确道路，因为这些自主系统在可靠性和透明度以及理解人类实际需求方面仍然存在问题。我们建议采取不同的方法：基于大语言模型的人机系统（LLM-HAS），其中AI与人类合作而不是取代他们。通过保留人类参与提供指导、回答问题和维持控制，这些系统可以更具信任度和适应性。通过医疗保健、金融和软件开发领域的例子，我们展示了人机合作如何比单独的AI更好地处理复杂任务。我们还讨论了构建这些协作系统的挑战，并提供了实用的解决方案。本文认为，人工智能的进步不应仅以其系统的独立性来衡量，而应以其与人类合作的能力来衡量。人工智能最有前途的未来不是在取代人类角色的系统，而是在通过有意义的合作增强人类能力的系统。 

---
# Beyond Nash Equilibrium: Bounded Rationality of LLMs and humans in Strategic Decision-making 

**Title (ZH)**: 超越纳什均衡：LLMs和人类在战略决策中的情境理性 

**Authors**: Kehan Zheng, Jinfeng Zhou, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09390)  

**Abstract**: Large language models are increasingly used in strategic decision-making settings, yet evidence shows that, like humans, they often deviate from full rationality. In this study, we compare LLMs and humans using experimental paradigms directly adapted from behavioral game-theory research. We focus on two well-studied strategic games, Rock-Paper-Scissors and the Prisoner's Dilemma, which are well known for revealing systematic departures from rational play in human subjects. By placing LLMs in identical experimental conditions, we evaluate whether their behaviors exhibit the bounded rationality characteristic of humans. Our findings show that LLMs reproduce familiar human heuristics, such as outcome-based strategy switching and increased cooperation when future interaction is possible, but they apply these rules more rigidly and demonstrate weaker sensitivity to the dynamic changes in the game environment. Model-level analyses reveal distinctive architectural signatures in strategic behavior, and even reasoning models sometimes struggle to find effective strategies in adaptive situations. These results indicate that current LLMs capture only a partial form of human-like bounded rationality and highlight the need for training methods that encourage flexible opponent modeling and stronger context awareness. 

**Abstract (ZH)**: 大规模语言模型在战略决策中的应用越来越多，但证据表明，与人类一样，它们往往偏离完全理性。本研究采用直接从行为博弈理论研究中adapted的实验范式，比较LLMs和人类的行为。我们重点关注两种已广泛研究的战略游戏——剪刀石头布和囚徒困境，这两种游戏因其揭示人类在理性行为方面系统性偏差而闻名。通过将LLMs置于相同的实验条件下，我们评估它们的行为是否表现出与人类类似的有限理性特征。研究发现，LLMs重现了经典的基于结果的战略切换和对未来互动可能性增加的合作行为，但它们应用这些规则更为僵化，并对外部游戏环境中的动态变化表现出较弱的敏感性。模型级分析揭示了战略行为中的独特架构特征，即便是推理模型有时也难以在适应性情境中找到有效的策略。这些结果表明，当前的LLMs仅捕获了人类有限理性的部分形式，并突显了需要通过训练方法鼓励灵活的对手建模和更强的上下文意识的必要性。 

---
# Ming-Omni: A Unified Multimodal Model for Perception and Generation 

**Title (ZH)**: 明-全知：统一的多模态感知与生成模型 

**Authors**: Inclusion AI, Biao Gong, Cheng Zou, Chuanyang Zheng, Chunluan Zhou, Canxiang Yan, Chunxiang Jin, Chunjie Shen, Dandan Zheng, Fudong Wang, Furong Xu, GuangMing Yao, Jun Zhou, Jingdong Chen, Jianxin Sun, Jiajia Liu, Jianjiang Zhu, Jun Peng, Kaixiang Ji, Kaiyou Song, Kaimeng Ren, Libin Wang, Lixiang Ru, Lele Xie, Longhua Tan, Lyuxin Xue, Lan Wang, Mochen Bai, Ning Gao, Pei Chen, Qingpei Guo, Qinglong Zhang, Qiang Xu, Rui Liu, Ruijie Xiong, Sirui Gao, Tinghao Liu, Taisong Li, Weilong Chai, Xinyu Xiao, Xiaomei Wang, Xiaoxue Chen, Xiao Lu, Xiaoyu Li, Xingning Dong, Xuzheng Yu, Yi Yuan, Yuting Gao, Yunxiao Sun, Yipeng Chen, Yifei Wu, Yongjie Lyu, Ziping Ma, Zipeng Feng, Zhijiang Fang, Zhihao Qiu, Ziyuan Huang, Zhengyu He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09344)  

**Abstract**: We propose Ming-Omni, a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-Omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-Omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-Omni offers a powerful solution for unified perception and generation across all modalities. Notably, our proposed Ming-Omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community. 

**Abstract (ZH)**: 我们提出Ming-Omni，一种统一的多模态模型，能够处理图像、文本、音频和视频，并在语音和图像生成方面展现出强大的能力。Ming-Omni采用专门的编码器从不同模态中提取令牌，这些令牌随后由装备有新提出的模态特定路由器的MoE架构Ling进行处理。这种设计使得单一模型能够在统一框架内高效地处理和融合多模态输入，从而实现在无需单独模型、任务特定微调或结构重设计的情况下完成多种任务。更重要的是，Ming-Omni超越了传统的多模态模型，支持音频和图像生成。这通过集成先进的音频解码器实现自然声音语音，并结合Ming-Lite-Uni进行高质量图像生成，使模型能够进行上下文感知聊天、文本转语音转换和多用途图像编辑。我们的实验结果展示了Ming-Omni提供了跨所有模态统一感知和生成的强大解决方案。值得注意的是，我们提出的Ming-Omni是我们所知的第一个开源模型，能够在模态支持方面与GPT-4o相媲美，我们发布了全部代码和模型权重，以鼓励社区进一步的研究和开发。 

---
# Comment on The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Title (ZH)**: 论思考的幻象：通过问题复杂性的视角理解推理模型的强弱之处 

**Authors**: C. Opus, A. Lawsen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09250)  

**Abstract**: Shojaee et al. (2025) report that Large Reasoning Models (LRMs) exhibit "accuracy collapse" on planning puzzles beyond certain complexity thresholds. We demonstrate that their findings primarily reflect experimental design limitations rather than fundamental reasoning failures. Our analysis reveals three critical issues: (1) Tower of Hanoi experiments systematically exceed model output token limits at reported failure points, with models explicitly acknowledging these constraints in their outputs; (2) The authors' automated evaluation framework fails to distinguish between reasoning failures and practical constraints, leading to misclassification of model capabilities; (3) Most concerningly, their River Crossing benchmarks include mathematically impossible instances for N > 5 due to insufficient boat capacity, yet models are scored as failures for not solving these unsolvable problems. When we control for these experimental artifacts, by requesting generating functions instead of exhaustive move lists, preliminary experiments across multiple models indicate high accuracy on Tower of Hanoi instances previously reported as complete failures. These findings highlight the importance of careful experimental design when evaluating AI reasoning capabilities. 

**Abstract (ZH)**: Shojaee等（2025）报告大型推理模型（LRMs）在超出一定复杂度阈值的规划谜题上表现出“准确性崩溃”。我们证明他们的发现主要反映了实验设计的局限性而非根本性的推理失败。我们的分析揭示了三个关键问题：（1）汉诺塔实验系统地在报告的失败点超过了模型输出的令牌限制，模型在输出中明确承认了这些约束；（2）作者的自动化评估框架无法区分推理失败和实际约束，导致对模型能力的误分类；（3）最令人担忧的是，他们的河流过河基准测试包括了对于N > 5来说数学上不可能的情况，由于小船容量不足，而模型因未能解决这些不可解的问题而被评分失败。当我们通过要求生成函数而非生成完整的移动列表来控制这些实验效应时，多个模型在之前报告为完全失败的汉诺塔实例上的初步实验显示出了高准确性。这些发现强调了评估人工智能推理能力时仔细实验设计的重要性。 

---
# Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism 

**Title (ZH)**: 机器人门控交互模仿学习与自适应干预机制 

**Authors**: Haoyuan Cai, Zhenghao Peng, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09176)  

**Abstract**: Interactive Imitation Learning (IIL) allows agents to acquire desired behaviors through human interventions, but current methods impose high cognitive demands on human supervisors. We propose the Adaptive Intervention Mechanism (AIM), a novel robot-gated IIL algorithm that learns an adaptive criterion for requesting human demonstrations. AIM utilizes a proxy Q-function to mimic the human intervention rule and adjusts intervention requests based on the alignment between agent and human actions. By assigning high Q-values when the agent deviates from the expert and decreasing these values as the agent becomes proficient, the proxy Q-function enables the agent to assess the real-time alignment with the expert and request assistance when needed. Our expert-in-the-loop experiments reveal that AIM significantly reduces expert monitoring efforts in both continuous and discrete control tasks. Compared to the uncertainty-based baseline Thrifty-DAgger, our method achieves a 40% improvement in terms of human take-over cost and learning efficiency. Furthermore, AIM effectively identifies safety-critical states for expert assistance, thereby collecting higher-quality expert demonstrations and reducing overall expert data and environment interactions needed. Code and demo video are available at this https URL. 

**Abstract (ZH)**: 自适应干预机制（AIM）：一种新颖的机器人门控imitation learning算法 

---
# DGS-LRM: Real-Time Deformable 3D Gaussian Reconstruction From Monocular Videos 

**Title (ZH)**: DGS-LRM: 基于单目视频的实时变形三维高斯重建 

**Authors**: Chieh Hubert Lin, Zhaoyang Lv, Songyin Wu, Zhen Xu, Thu Nguyen-Phuoc, Hung-Yu Tseng, Julian Straub, Numair Khan, Lei Xiao, Ming-Hsuan Yang, Yuheng Ren, Richard Newcombe, Zhao Dong, Zhengqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09997)  

**Abstract**: We introduce the Deformable Gaussian Splats Large Reconstruction Model (DGS-LRM), the first feed-forward method predicting deformable 3D Gaussian splats from a monocular posed video of any dynamic scene. Feed-forward scene reconstruction has gained significant attention for its ability to rapidly create digital replicas of real-world environments. However, most existing models are limited to static scenes and fail to reconstruct the motion of moving objects. Developing a feed-forward model for dynamic scene reconstruction poses significant challenges, including the scarcity of training data and the need for appropriate 3D representations and training paradigms. To address these challenges, we introduce several key technical contributions: an enhanced large-scale synthetic dataset with ground-truth multi-view videos and dense 3D scene flow supervision; a per-pixel deformable 3D Gaussian representation that is easy to learn, supports high-quality dynamic view synthesis, and enables long-range 3D tracking; and a large transformer network that achieves real-time, generalizable dynamic scene reconstruction. Extensive qualitative and quantitative experiments demonstrate that DGS-LRM achieves dynamic scene reconstruction quality comparable to optimization-based methods, while significantly outperforming the state-of-the-art predictive dynamic reconstruction method on real-world examples. Its predicted physically grounded 3D deformation is accurate and can readily adapt for long-range 3D tracking tasks, achieving performance on par with state-of-the-art monocular video 3D tracking methods. 

**Abstract (ZH)**: 可变形高斯点大规模重建模型 (DGS-LRM): 从任意动态场景的单目视频预测可变形 3D 高斯点的首个前馈方法 

---
# eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures 

**Title (ZH)**: eFlesh: 高度可定制的切割细胞微结构磁触感技术 

**Authors**: Venkatesh Pattabiraman, Zizhou Huang, Daniele Panozzo, Denis Zorin, Lerrel Pinto, Raunaq Bhirangi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09994)  

**Abstract**: If human experience is any guide, operating effectively in unstructured environments -- like homes and offices -- requires robots to sense the forces during physical interaction. Yet, the lack of a versatile, accessible, and easily customizable tactile sensor has led to fragmented, sensor-specific solutions in robotic manipulation -- and in many cases, to force-unaware, sensorless approaches. With eFlesh, we bridge this gap by introducing a magnetic tactile sensor that is low-cost, easy to fabricate, and highly customizable. Building an eFlesh sensor requires only four components: a hobbyist 3D printer, off-the-shelf magnets (<$5), a CAD model of the desired shape, and a magnetometer circuit board. The sensor is constructed from tiled, parameterized microstructures, which allow for tuning the sensor's geometry and its mechanical response. We provide an open-source design tool that converts convex OBJ/STL files into 3D-printable STLs for fabrication. This modular design framework enables users to create application-specific sensors, and to adjust sensitivity depending on the task. Our sensor characterization experiments demonstrate the capabilities of eFlesh: contact localization RMSE of 0.5 mm, and force prediction RMSE of 0.27 N for normal force and 0.12 N for shear force. We also present a learned slip detection model that generalizes to unseen objects with 95% accuracy, and visuotactile control policies that improve manipulation performance by 40% over vision-only baselines -- achieving 91% average success rate for four precise tasks that require sub-mm accuracy for successful completion. All design files, code and the CAD-to-eFlesh STL conversion tool are open-sourced and available on this https URL. 

**Abstract (ZH)**: 基于eFlesh的低成本可定制磁性触觉传感器及其应用 

---
# Text-Aware Image Restoration with Diffusion Models 

**Title (ZH)**: 基于文本的图像恢复扩散模型 

**Authors**: Jaewon Min, Jin Hyeon Kim, Paul Hyunbin Cho, Jaeeun Lee, Jihye Park, Minkyu Park, Sangpil Kim, Hyunhee Park, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09993)  

**Abstract**: Image restoration aims to recover degraded images. However, existing diffusion-based restoration methods, despite great success in natural image restoration, often struggle to faithfully reconstruct textual regions in degraded images. Those methods frequently generate plausible but incorrect text-like patterns, a phenomenon we refer to as text-image hallucination. In this paper, we introduce Text-Aware Image Restoration (TAIR), a novel restoration task that requires the simultaneous recovery of visual contents and textual fidelity. To tackle this task, we present SA-Text, a large-scale benchmark of 100K high-quality scene images densely annotated with diverse and complex text instances. Furthermore, we propose a multi-task diffusion framework, called TeReDiff, that integrates internal features from diffusion models into a text-spotting module, enabling both components to benefit from joint training. This allows for the extraction of rich text representations, which are utilized as prompts in subsequent denoising steps. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art restoration methods, achieving significant gains in text recognition accuracy. See our project page: this https URL 

**Abstract (ZH)**: 图像恢复旨在恢复退化图像。然而，现有的基于扩散的恢复方法尽管在自然图像恢复方面取得了巨大成功，但在重建退化图像中的文本区域时往往难以做到忠实重构。这些方法经常生成合理但错误的文字样图案，我们将这一现象称为文本图像幻视。本文介绍了一种新的恢复任务——文本感知图像恢复（TAIR），该任务要求同时恢复视觉内容和文本保真度。为应对这一任务，我们提出了SA-Text，这是一个包含10万张高质量场景图像的大规模基准数据集，这些图像密集标注了多样且复杂的文本实例。此外，我们提出了一种多任务扩散框架TeReDiff，将扩散模型的内部特征整合到一个文本检测模块中，使两个组件能够从联合训练中受益，从而提取丰富的文本表示，这些表示用于后续去噪步骤中作为提示。大量实验表明，我们的方法在文本识别准确性方面始终优于最先进的恢复方法，取得了显著的提升。见我们的项目页面：this https URL。 

---
# EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits 

**Title (ZH)**: 文本引导图像编辑评估基准：EditInspector 

**Authors**: Ron Yosef, Moran Yanuka, Yonatan Bitton, Dani Lischinski  

**Link**: [PDF](https://arxiv.org/pdf/2506.09988)  

**Abstract**: Text-guided image editing, fueled by recent advancements in generative AI, is becoming increasingly widespread. This trend highlights the need for a comprehensive framework to verify text-guided edits and assess their quality. To address this need, we introduce EditInspector, a novel benchmark for evaluation of text-guided image edits, based on human annotations collected using an extensive template for edit verification. We leverage EditInspector to evaluate the performance of state-of-the-art (SoTA) vision and language models in assessing edits across various dimensions, including accuracy, artifact detection, visual quality, seamless integration with the image scene, adherence to common sense, and the ability to describe edit-induced changes. Our findings indicate that current models struggle to evaluate edits comprehensively and frequently hallucinate when describing the changes. To address these challenges, we propose two novel methods that outperform SoTA models in both artifact detection and difference caption generation. 

**Abstract (ZH)**: 基于文本引导的图像编辑：随着生成AI的 recent 进展日益普及，需构建全面框架进行验证与评估 

---
# InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions 

**Title (ZH)**: InterActHuman：布局对齐音频条件驱动的多概念人体动画生成 

**Authors**: Zhenzhi Wang, Jiaqi Yang, Jianwen Jiang, Chao Liang, Gaojie Lin, Zerong Zheng, Ceyuan Yang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09984)  

**Abstract**: End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios that multiple concepts could appears in the same video with rich human-human interactions and human-object interactions. Such global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region-specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in a iterative manner. This design enables the high-quality generation of controllable multi-concept human-centric videos. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods. 

**Abstract (ZH)**: 端到端多模态条件下的丰富人体动画：摆脱单一主体假设，实现精确的身份特定控制 

---
# Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing 

**Title (ZH)**: 在交织思考与视觉绘制中强化视觉语言模型的空间推理能力 

**Authors**: Junfei Wu, Jian Guan, Kaituo Feng, Qiang Liu, Shu Wu, Liang Wang, Wei Wu, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09965)  

**Abstract**: As textual reasoning with large language models (LLMs) has advanced significantly, there has been growing interest in enhancing the multimodal reasoning capabilities of large vision-language models (LVLMs). However, existing methods primarily approach multimodal reasoning in a straightforward, text-centric manner, where both reasoning and answer derivation are conducted purely through text, with the only difference being the presence of multimodal input. As a result, these methods often encounter fundamental limitations in spatial reasoning tasks that demand precise geometric understanding and continuous spatial tracking-capabilities that humans achieve through mental visualization and manipulation. To address the limitations, we propose drawing to reason in space, a novel paradigm that enables LVLMs to reason through elementary drawing operations in the visual space. By equipping models with basic drawing operations, including annotating bounding boxes and drawing auxiliary lines, we empower them to express and analyze spatial relationships through direct visual manipulation, meanwhile avoiding the performance ceiling imposed by specialized perception tools in previous tool-integrated reasoning approaches. To cultivate this capability, we develop a three-stage training framework: cold-start training with synthetic data to establish basic drawing abilities, reflective rejection sampling to enhance self-reflection behaviors, and reinforcement learning to directly optimize for target rewards. Extensive experiments demonstrate that our model, named VILASR, consistently outperforms existing methods across diverse spatial reasoning benchmarks, involving maze navigation, static spatial reasoning, video-based reasoning, and multi-view-based reasoning tasks, with an average improvement of 18.4%. 

**Abstract (ZH)**: 基于空间绘图的视觉语言模型空间推理新范式 

---
# LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge 

**Title (ZH)**: LLMail-Inject：一项现实适应性提示注入挑战的数据集 

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09956)  

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection. 

**Abstract (ZH)**: 间接提示注入攻击利用了大规模语言模型（LLMs）区分输入中的指令与数据的固有限制。尽管提出了许多防御方案，针对适应性对手的系统评估仍然有限，即使成功的攻击可能具有广泛的安全和隐私影响，许多基于LLM的实际应用仍然易受攻击。我们介绍了LLMail-Inject公挑战的结果，该挑战模拟了一个现实场景，在该场景中，参与者适应性地尝试将恶意指令注入电子邮件，以触发基于LLM的电子邮件助手的未授权工具调用。该挑战涵盖了多种防御策略、LLM架构和检索配置，产生了包含839名参与者208,095个独特攻击提交的数据集。我们发布了挑战代码、完整的提交数据集以及我们的分析，展示了这些数据如何提供有关指令-数据区分问题的新见解。我们希望这将为未来针对提示注入的实际结构解决方案的研究奠定基础。 

---
# Vision Generalist Model: A Survey 

**Title (ZH)**: 视觉通用模型：一篇综述 

**Authors**: Ziyi Wang, Yongming Rao, Shuofeng Sun, Xinrun Liu, Yi Wei, Xumin Yu, Zuyan Liu, Yanbo Wang, Hongmin Liu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09954)  

**Abstract**: Recently, we have witnessed the great success of the generalist model in natural language processing. The generalist model is a general framework trained with massive data and is able to process various downstream tasks simultaneously. Encouraged by their impressive performance, an increasing number of researchers are venturing into the realm of applying these models to computer vision tasks. However, the inputs and outputs of vision tasks are more diverse, and it is difficult to summarize them as a unified representation. In this paper, we provide a comprehensive overview of the vision generalist models, delving into their characteristics and capabilities within the field. First, we review the background, including the datasets, tasks, and benchmarks. Then, we dig into the design of frameworks that have been proposed in existing research, while also introducing the techniques employed to enhance their performance. To better help the researchers comprehend the area, we take a brief excursion into related domains, shedding light on their interconnections and potential synergies. To conclude, we provide some real-world application scenarios, undertake a thorough examination of the persistent challenges, and offer insights into possible directions for future research endeavors. 

**Abstract (ZH)**: 最近，通用模型在自然语言处理领域取得了巨大成功。通用模型是一种通过大规模数据训练的通用框架，能够同时处理多种下游任务。受到其出色表现的鼓舞，越来越多的研究人员将这些模型应用于计算机视觉任务。然而，视觉任务的输入和输出更为多样化，难以将其统一概括。本文对视觉通用模型进行了全面概述，探讨了其在该领域的特点和能力。首先，我们回顾了背景信息，包括数据集、任务和基准测试。接着，我们深入探讨了现有研究中提出的框架设计，并介绍了增强其性能的技术。为了更好地帮助研究人员理解该领域，我们对相关领域进行了简要的调研，揭示了它们之间的联系和潜在协同效应。最后，我们提供了实际应用场景，详细分析了持续存在的挑战，并提出了未来研究方向的见解。 

---
# Outside Knowledge Conversational Video (OKCV) Dataset -- Dialoguing over Videos 

**Title (ZH)**: Outside Knowledge Conversational Video (OKCV) 数据集 -- 视频上的对话 

**Authors**: Benjamin Reichman, Constantin Patsch, Jack Truxal, Atishay Jain, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2506.09953)  

**Abstract**: In outside knowledge visual question answering (OK-VQA), the model must identify relevant visual information within an image and incorporate external knowledge to accurately respond to a question. Extending this task to a visually grounded dialogue setting based on videos, a conversational model must both recognize pertinent visual details over time and answer questions where the required information is not necessarily present in the visual information. Moreover, the context of the overall conversation must be considered for the subsequent dialogue. To explore this task, we introduce a dataset comprised of $2,017$ videos with $5,986$ human-annotated dialogues consisting of $40,954$ interleaved dialogue turns. While the dialogue context is visually grounded in specific video segments, the questions further require external knowledge that is not visually present. Thus, the model not only has to identify relevant video parts but also leverage external knowledge to converse within the dialogue. We further provide several baselines evaluated on our dataset and show future challenges associated with this task. The dataset is made publicly available here: this https URL. 

**Abstract (ZH)**: 在外包知识视觉问答（OK-VQA）中，模型必须识别图像内的相关视觉信息并结合外部知识以准确回答问题。将此任务扩展到基于视频的对话设置中，对话模型不仅要随着时间识别相关视觉细节，还要回答那些所需信息不一定在视觉信息中出现的问题。此外，整个对话的上下文必须被考虑以供后续对话使用。为了探索这一任务，我们引入了一个包含2017个视频和5986个人工标注对话的数据集，共计40954个对话回合。尽管对话的上下文是基于特定视频片段的视觉 grounding，但问题进一步要求外部知识，而这在视觉信息中未出现。因此，模型不仅要识别相关视频部分，还要利用外部知识进行对话。我们还提供了在该数据集上评估的几种基线，并展示了与此任务相关的未来挑战。该数据集已公开发布于此：this https URL。 

---
# UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting 

**Title (ZH)**: UniPre3D：统一的3D点云模型跨模态高斯绘制预训练 

**Authors**: Ziyi Wang, Yanran Zhang, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09952)  

**Abstract**: The scale diversity of point cloud data presents significant challenges in developing unified representation learning techniques for 3D vision. Currently, there are few unified 3D models, and no existing pre-training method is equally effective for both object- and scene-level point clouds. In this paper, we introduce UniPre3D, the first unified pre-training method that can be seamlessly applied to point clouds of any scale and 3D models of any architecture. Our approach predicts Gaussian primitives as the pre-training task and employs differentiable Gaussian splatting to render images, enabling precise pixel-level supervision and end-to-end optimization. To further regulate the complexity of the pre-training task and direct the model's focus toward geometric structures, we integrate 2D features from pre-trained image models to incorporate well-established texture knowledge. We validate the universal effectiveness of our proposed method through extensive experiments across a variety of object- and scene-level tasks, using diverse point cloud models as backbones. Code is available at this https URL. 

**Abstract (ZH)**: 点云数据的尺度多样性给开发统一的3D视觉表示学习技术带来了显著挑战。目前，很少存在统一的3D模型，且现有预训练方法在对象级和场景级点云上均未表现出同等效用。在这项工作中，我们引入了UniPre3D，这是首个能够无缝应用于任意尺度点云和任意架构3D模型的统一预训练方法。我们的方法以预测高斯基础元作为预训练任务，并采用可微分的高斯插值生成图像，从而提供精确的像素级监督和端到端优化。为进一步调节预训练任务的复杂性并引导模型关注几何结构，我们将预训练图像模型的2D特征整合进来，以引入成熟的空间纹理知识。我们通过广泛实验验证了所提方法在多种对象级和场景级任务上的通用有效性，使用了多种点云模型作为骨干。代码详见这个链接。 

---
# CausalVQA: A Physically Grounded Causal Reasoning Benchmark for Video Models 

**Title (ZH)**: 因果VQA：面向视频模型的物理基础因果推理基准 

**Authors**: Aaron Foss, Chloe Evans, Sasha Mitts, Koustuv Sinha, Ammar Rizvi, Justine T. Kao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09943)  

**Abstract**: We introduce CausalVQA, a benchmark dataset for video question answering (VQA) composed of question-answer pairs that probe models' understanding of causality in the physical world. Existing VQA benchmarks either tend to focus on surface perceptual understanding of real-world videos, or on narrow physical reasoning questions created using simulation environments. CausalVQA fills an important gap by presenting challenging questions that are grounded in real-world scenarios, while focusing on models' ability to predict the likely outcomes of different actions and events through five question types: counterfactual, hypothetical, anticipation, planning and descriptive. We designed quality control mechanisms that prevent models from exploiting trivial shortcuts, requiring models to base their answers on deep visual understanding instead of linguistic cues. We find that current frontier multimodal models fall substantially below human performance on the benchmark, especially on anticipation and hypothetical questions. This highlights a challenge for current systems to leverage spatial-temporal reasoning, understanding of physical principles, and comprehension of possible alternatives to make accurate predictions in real-world settings. 

**Abstract (ZH)**: 因果关系视觉问答基准数据集：探究模型在物理世界中因果关系理解的能力 

---
# VerIF: Verification Engineering for Reinforcement Learning in Instruction Following 

**Title (ZH)**: VerIF: 强化学习在指令遵循中 struggl 工程验证 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09942)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a key technique for enhancing large language models (LLMs), with verification engineering playing a central role. However, best practices for RL in instruction following remain underexplored. In this work, we explore the verification challenge in RL for instruction following and propose VerIF, a verification method that combines rule-based code verification with LLM-based verification from a large reasoning model (e.g., QwQ-32B). To support this approach, we construct a high-quality instruction-following dataset, VerInstruct, containing approximately 22,000 instances with associated verification signals. We apply RL training with VerIF to two models, achieving significant improvements across several representative instruction-following benchmarks. The trained models reach state-of-the-art performance among models of comparable size and generalize well to unseen constraints. We further observe that their general capabilities remain unaffected, suggesting that RL with VerIF can be integrated into existing RL recipes to enhance overall model performance. We have released our datasets, codes, and models to facilitate future research at this https URL. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已成为提升大规模语言模型（LLMs）的关键技术，验证工程起着核心作用。然而，指令跟随中的强化学习最佳实践仍待探索。本工作中，我们探讨了指令跟随中强化学习的验证挑战，并提出了一种结合规则基于的代码验证和大型推理模型（如QwQ-32B）基于的验证方法VerIF。为支持该方法，我们构建了一个高质量的指令跟随数据集VerInstruct，包含约22,000个实例及其关联的验证信号。我们将VerIF应用于两个模型，并在多个代表性指令跟随基准测试中取得了显著的改进。训练后的模型在可比较规模的模型中达到最佳性能，并且能够在未见过的约束下良好泛化。此外，我们观察到其通用能力未受影响，表明带有VerIF的强化学习可以集成到现有的强化学习配方中以提升整体模型性能。我们已公开了数据集、代码和模型以方便未来研究。 

---
# The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability 

**Title (ZH)**: 具有信息不对称和知识可转移性的在线战略决策的样本复杂性 

**Authors**: Jiachen Hu, Rui Ai, Han Zhong, Xiaoyu Chen, Liwei Wang, Zhaoran Wang, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09940)  

**Abstract**: Information asymmetry is a pervasive feature of multi-agent systems, especially evident in economics and social sciences. In these settings, agents tailor their actions based on private information to maximize their rewards. These strategic behaviors often introduce complexities due to confounding variables. Simultaneously, knowledge transportability poses another significant challenge, arising from the difficulties of conducting experiments in target environments. It requires transferring knowledge from environments where empirical data is more readily available. Against these backdrops, this paper explores a fundamental question in online learning: Can we employ non-i.i.d. actions to learn about confounders even when requiring knowledge transfer? We present a sample-efficient algorithm designed to accurately identify system dynamics under information asymmetry and to navigate the challenges of knowledge transfer effectively in reinforcement learning, framed within an online strategic interaction model. Our method provably achieves learning of an $\epsilon$-optimal policy with a tight sample complexity of $O(1/\epsilon^2)$. 

**Abstract (ZH)**: 信息不对称是多Agent系统中一个普遍特征，尤其是在经济学和社会科学领域中尤为明显。在这种环境中，代理基于私有信息调整其行为以最大化奖励。这些战略性行为常常由于混杂变量的引入而增加复杂性。同时，知识可转移性也提出了另一个重大挑战，源于在目标环境中进行实验的困难。这需要从更容易获得经验数据的环境中转移知识。在此背景下，本文探讨了在线学习中的一个基础问题：我们是否可以使用非独立同分布（non-i.i.d.）的动作来学习混杂变量，即使在这种情况下需要进行知识转移？我们提出了一种样本高效的算法，旨在在信息不对称条件下准确识别系统动力学，并有效地在强化学习框架内的在线战略交互模型中应对知识转移的挑战。我们的方法能够证明以紧致的样本复杂度$O(1/\epsilon^2)$学习$\epsilon$-最优策略。 

---
# SAFE: Multitask Failure Detection for Vision-Language-Action Models 

**Title (ZH)**: SAFE：面向视觉-语言-行动模型的多任务故障检测 

**Authors**: Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.09937)  

**Abstract**: While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out-of-the-box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient high-level knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts, and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, $\pi_0$, and $\pi_0$-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and the best trade-off between accuracy and detection time using conformal prediction. More qualitative results can be found at this https URL. 

**Abstract (ZH)**: 多任务故障检测：面向通用机器人策略的安全检测器 

---
# HadaNorm: Diffusion Transformer Quantization through Mean-Centered Transformations 

**Title (ZH)**: HadaNorm：基于均值中心化变换的扩散变压器量化 

**Authors**: Marco Federici, Riccardo Del Chiaro, Boris van Breugel, Paul Whatmough, Markus Nagel  

**Link**: [PDF](https://arxiv.org/pdf/2506.09932)  

**Abstract**: Diffusion models represent the cutting edge in image generation, but their high memory and computational demands hinder deployment on resource-constrained devices. Post-Training Quantization (PTQ) offers a promising solution by reducing the bitwidth of matrix operations. However, standard PTQ methods struggle with outliers, and achieving higher compression often requires transforming model weights and activations before quantization. In this work, we propose HadaNorm, a novel linear transformation that extends existing approaches and effectively mitigates outliers by normalizing activations feature channels before applying Hadamard transformations, enabling more aggressive activation quantization. We demonstrate that HadaNorm consistently reduces quantization error across the various components of transformer blocks, achieving superior efficiency-performance trade-offs when compared to state-of-the-art methods. 

**Abstract (ZH)**: 基于HadaNorm的新型线性变换在变压器块中有效缓解异常值，实现更优的效率-性能 trade-offs 

---
# PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants 

**Title (ZH)**: PersonaLens：面向对话AI助手个性化评价的标准基准 

**Authors**: Zheng Zhao, Clara Vania, Subhradeep Kayal, Naila Khan, Shay B. Cohen, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.09902)  

**Abstract**: Large language models (LLMs) have advanced conversational AI assistants. However, systematically evaluating how well these assistants apply personalization--adapting to individual user preferences while completing tasks--remains challenging. Existing personalization benchmarks focus on chit-chat, non-conversational tasks, or narrow domains, failing to capture the complexities of personalized task-oriented assistance. To address this, we introduce PersonaLens, a comprehensive benchmark for evaluating personalization in task-oriented AI assistants. Our benchmark features diverse user profiles equipped with rich preferences and interaction histories, along with two specialized LLM-based agents: a user agent that engages in realistic task-oriented dialogues with AI assistants, and a judge agent that employs the LLM-as-a-Judge paradigm to assess personalization, response quality, and task success. Through extensive experiments with current LLM assistants across diverse tasks, we reveal significant variability in their personalization capabilities, providing crucial insights for advancing conversational AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）已推动了对话式人工智能助手的发展。然而，系统地评估这些助手在个性化方面的能力——即根据不同用户偏好完成任务的能力——仍具有挑战性。现有的个性化基准主要集中在闲聊、非对话任务或狭窄领域，未能捕捉到个性化任务导向辅助的复杂性。为解决这一问题，我们引入了PersonaLens，一个全面的基准测试，用于评估任务导向人工智能助手的个性化能力。该基准包括多样化的用户画像，配备了丰富的偏好和交互历史，并配备了两个专门的LLM基代理：一个用户代理，与人工智能助手进行真实的任务导向对话；一个评判代理，采用LLM作为评判者的范式，评估个性化、响应质量和任务完成情况。通过广泛的实验，我们在多种任务中测评当前的LLM助手，揭示了他们在个性化能力方面的显著差异，提供了对推进对话式人工智能系统的重要见解。 

---
# Causal Climate Emulation with Bayesian Filtering 

**Title (ZH)**: 基于贝叶斯滤波的因果气候模拟 

**Authors**: Sebastian Hickman, Ilija Trajkovic, Julia Kaltenborn, Francis Pelletier, Alex Archibald, Yaniv Gurwicz, Peer Nowack, David Rolnick, Julien Boussard  

**Link**: [PDF](https://arxiv.org/pdf/2506.09891)  

**Abstract**: Traditional models of climate change use complex systems of coupled equations to simulate physical processes across the Earth system. These simulations are highly computationally expensive, limiting our predictions of climate change and analyses of its causes and effects. Machine learning has the potential to quickly emulate data from climate models, but current approaches are not able to incorporate physics-informed causal relationships. Here, we develop an interpretable climate model emulator based on causal representation learning. We derive a physics-informed approach including a Bayesian filter for stable long-term autoregressive emulation. We demonstrate that our emulator learns accurate climate dynamics, and we show the importance of each one of its components on a realistic synthetic dataset and data from two widely deployed climate models. 

**Abstract (ZH)**: 基于因果表示学习的可解释气候模型仿真器：包含物理信息的稳定长期自回归仿真 

---
# The Emergence of Abstract Thought in Large Language Models Beyond Any Language 

**Title (ZH)**: 大型语言模型中抽象思维的 emergence 超越语言本身的界限 

**Authors**: Yuxin Chen, Yiran Zhao, Yang Zhang, An Zhang, Kenji Kawaguchi, Shafiq Joty, Junnan Li, Tat-Seng Chua, Michael Qizhe Shieh, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09890)  

**Abstract**: As large language models (LLMs) continue to advance, their capacity to function effectively across a diverse range of languages has shown marked improvement. Preliminary studies observe that the hidden activations of LLMs often resemble English, even when responding to non-English prompts. This has led to the widespread assumption that LLMs may "think" in English. However, more recent results showing strong multilingual performance, even surpassing English performance on specific tasks in other languages, challenge this view. In this work, we find that LLMs progressively develop a core language-agnostic parameter space-a remarkably small subset of parameters whose deactivation results in significant performance degradation across all languages. This compact yet critical set of parameters underlies the model's ability to generalize beyond individual languages, supporting the emergence of abstract thought that is not tied to any specific linguistic system. Specifically, we identify language-related neurons-those are consistently activated during the processing of particular languages, and categorize them as either shared (active across multiple languages) or exclusive (specific to one). As LLMs undergo continued development over time, we observe a marked increase in both the proportion and functional importance of shared neurons, while exclusive neurons progressively diminish in influence. These shared neurons constitute the backbone of the core language-agnostic parameter space, supporting the emergence of abstract thought. Motivated by these insights, we propose neuron-specific training strategies tailored to LLMs' language-agnostic levels at different development stages. Experiments across diverse LLM families support our approach. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的不断进步，它们在多样语言范围内的有效运行能力得到了显著提升。初步研究观察到，LLMs在回应非英语提示时，其隐藏激活状态往往类似于英语。这导致广泛假设LLMs可能会“以英语思考”。然而，近期显示出强大多语言性能的结果，甚至在某些任务上超越了单独语言的英语性能，挑战了这一观点。在本工作中，我们发现LLMs逐渐发展出一个核心语言无关的参数空间——一个极其小的参数子集，其失活会导致所有语言性能显著下降。这一紧凑但关键的参数集合支撑了模型在单一语言之外的泛化能力，促进了不依赖任何特定语言系统的抽象思维的出现。具体而言，我们识别出与特定语言处理过程中一致激活的语言相关的神经元，并将其分类为共享（跨多种语言活跃）或专属（仅针对一种语言）。随着LLMs不断发展，我们观察到共享神经元的比例和功能重要性显著增加，而专属神经元的影响则逐渐减弱。这些共享神经元构成核心语言无关参数空间的骨干，支持抽象思维的出现。受此启发，我们提出针对LLMs在不同发展阶段的语言无关水平定制的神经元特定训练策略。来自多样化LLM家族的实验支持了我们的方法。 

---
# Attention Head Embeddings with Trainable Deep Kernels for Hallucination Detection in LLMs 

**Title (ZH)**: 基于可训练深度内核的注意力头嵌入在大语言模型中检测幻觉 

**Authors**: Rodion Oblovatny, Alexandra Bazarova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.09886)  

**Abstract**: We present a novel approach for detecting hallucinations in large language models (LLMs) by analyzing the probabilistic divergence between prompt and response hidden-state distributions. Counterintuitively, we find that hallucinated responses exhibit smaller deviations from their prompts compared to grounded responses, suggesting that hallucinations often arise from superficial rephrasing rather than substantive reasoning. Leveraging this insight, we propose a model-intrinsic detection method that uses distributional distances as principled hallucination scores, eliminating the need for external knowledge or auxiliary models. To enhance sensitivity, we employ deep learnable kernels that automatically adapt to capture nuanced geometric differences between distributions. Our approach outperforms existing baselines, demonstrating state-of-the-art performance on several benchmarks. The method remains competitive even without kernel training, offering a robust, scalable solution for hallucination detection. 

**Abstract (ZH)**: 我们提出了一种通过分析提示和响应隐藏状态分布的概率性差异来检测大型语言模型中幻觉的新方法。令人意外的是，我们发现幻觉响应与提示之间的偏差小于基于事实的响应，这表明幻觉通常源自表面的改述而非实质性的推理。基于这一见解，我们提出了一种模型固有的检测方法，利用分布距离作为幻觉评分，从而消除对外部知识或辅助模型的依赖。为了提高灵敏度，我们采用了可学习的深内核，能够自动适应捕捉分布之间微妙的几何差异。我们的方法在多个基准测试上优于现有基线，展示了最先进的性能。即使不进行内核训练，该方法仍具有竞争力，提供了一种稳健且可扩展的幻觉检测解决方案。 

---
# 3D-Aware Vision-Language Models Fine-Tuning with Geometric Distillation 

**Title (ZH)**: 三维意识视觉-语言模型几何蒸馏 fine-tuning 

**Authors**: Seonho Lee, Jiho Choi, Inha Kang, Jiwook Kim, Junsung Park, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09883)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable performance on diverse visual and linguistic tasks, yet they remain fundamentally limited in their understanding of 3D spatial structures. We propose Geometric Distillation, a lightweight, annotation-free fine-tuning framework that injects human-inspired geometric cues into pretrained VLMs without modifying their architecture. By distilling (1) sparse correspondences, (2) relative depth relations, and (3) dense cost volumes from off-the-shelf 3D foundation models (e.g., MASt3R, VGGT), our method shapes representations to be geometry-aware while remaining compatible with natural image-text inputs. Through extensive evaluations on 3D vision-language reasoning and 3D perception benchmarks, our method consistently outperforms prior approaches, achieving improved 3D spatial reasoning with significantly lower computational cost. Our work demonstrates a scalable and efficient path to bridge 2D-trained VLMs with 3D understanding, opening up wider use in spatially grounded multimodal tasks. 

**Abstract (ZH)**: 几何蒸馏：一种轻量级、无注释的细调框架，将人类启发的几何线索注入预训练的视觉-语言模型以增强三维空间理解 

---
# Stakeholder Participation for Responsible AI Development: Disconnects Between Guidance and Current Practice 

**Title (ZH)**: 负责任人工智能开发中的利益相关者参与：指南与当前实践之间的差距 

**Authors**: Emma Kallina, Thomas Bohné, Jat Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.09873)  

**Abstract**: Responsible AI (rAI) guidance increasingly promotes stakeholder involvement (SHI) during AI development. At the same time, SHI is already common in commercial software development, but with potentially different foci. This study clarifies the extent to which established SHI practices are able to contribute to rAI efforts as well as potential disconnects -- essential insights to inform and tailor future interventions that further shift industry practice towards rAI efforts. First, we analysed 56 rAI guidance documents to identify why SHI is recommended (i.e. its expected benefits for rAI) and uncovered goals such as redistributing power, improving socio-technical understandings, anticipating risks, and enhancing public oversight. To understand why and how SHI is currently practised in commercial settings, we then conducted an online survey (n=130) and semi-structured interviews (n=10) with AI practitioners. Our findings reveal that SHI in practice is primarily driven by commercial priorities (e.g. customer value, compliance) and several factors currently discourage more rAI-aligned SHI practices. This suggests that established SHI practices are largely not contributing to rAI efforts. To address this disconnect, we propose interventions and research opportunities to advance rAI development in practice. 

**Abstract (ZH)**: 负责任人工智能（rAI）指导越来越强调在人工智能开发过程中增加利益相关者参与（SHI）。与此同时，利益相关者参与已经在商业软件开发中普遍存在，但可能侧重不同。本研究阐明了现有SHI实践在多大程度上能够促进rAI努力，以及潜在的脱节——这些见解对于指导和定制未来进一步推动行业实践向rAI努力的干预措施至关重要。首先，我们分析了56份rAI指导文档，以确定推荐SHI的原因（即其对rAI的预期益处），并发现了一些目标，如重新分配权力、改善社会技术理解、预见风险和增强公众监督。为了了解商业环境中当前的SHI实践为何以及如何进行，我们随后对130名AI从业者进行了在线调查，并对10名从业者进行了半结构化访谈。我们的研究发现，在实践中，SHI主要由商业优先事项（如客户价值、合规性）驱动，当前有多种因素阻碍了更符合rAI的SHI实践。这表明现有SHI实践并未大量贡献于rAI努力。为了解决这一脱节，我们提出了推进rAI开发的干预措施和研究机会。 

---
# Guided Graph Compression for Quantum Graph Neural Networks 

**Title (ZH)**: 引导图压缩用于量子图神经网络 

**Authors**: Mikel Casals, Vasilis Belis, Elias F. Combarro, Eduard Alarcón, Sofia Vallecorsa, Michele Grossi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09862)  

**Abstract**: Graph Neural Networks (GNNs) are effective for processing graph-structured data but face challenges with large graphs due to high memory requirements and inefficient sparse matrix operations on GPUs. Quantum Computing (QC) offers a promising avenue to address these issues and inspires new algorithmic approaches. In particular, Quantum Graph Neural Networks (QGNNs) have been explored in recent literature. However, current quantum hardware limits the dimension of the data that can be effectively encoded. Existing approaches either simplify datasets manually or use artificial graph datasets. This work introduces the Guided Graph Compression (GGC) framework, which uses a graph autoencoder to reduce both the number of nodes and the dimensionality of node features. The compression is guided to enhance the performance of a downstream classification task, which can be applied either with a quantum or a classical classifier. The framework is evaluated on the Jet Tagging task, a classification problem of fundamental importance in high energy physics that involves distinguishing particle jets initiated by quarks from those by gluons. The GGC is compared against using the autoencoder as a standalone preprocessing step and against a baseline classical GNN classifier. Our numerical results demonstrate that GGC outperforms both alternatives, while also facilitating the testing of novel QGNN ansatzes on realistic datasets. 

**Abstract (ZH)**: Graph神经网络（GNNs）在处理图结构数据方面效果显著，但在大规模图上面临高内存需求和不高效的稀疏矩阵操作问题。量子计算（QC）提供了一种有前景的方法来解决这些问题，并启发了新的算法方法。特别是，近期文献中探讨了量子图神经网络（QGNNs）。然而，当前的量子硬件限制了能有效编码的数据维度。现有方法要么手动简化数据集，要么使用人工生成的图数据集。本文引入了Guided图压缩（GGC）框架，该框架使用图自编码器减少节点数和节点特征的维度。压缩过程受到后续分类任务性能的指导，可以结合量子或经典分类器应用。该框架在高能物理中至关重要的喷流标记任务上进行了评估，该任务涉及区分由夸克和由胶子引起的喷流。GGC与仅作为独立预处理步骤的自编码器以及基准经典GNN分类器进行了比较。我们的数值结果表明，GGC在性能上优于两种替代方案，同时也为在真实数据集上测试新的QGNN变体提供了途径。 

---
# Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning 

**Title (ZH)**: 因果充分性和必要性提高链式思维推理 

**Authors**: Xiangning Yu, Zhuohan Wang, Linyi Yang, Haoxuan Li, Anjie Liu, Xiao Xue, Jun Wang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09853)  

**Abstract**: Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness. 

**Abstract (ZH)**: 因果框架通过充足性和必要性的双重视角 karakterize CoT推理，结合因果概率的充足性和必要性，不仅可以确定哪些步骤对预测结果是逻辑上充分或必要的，还可以在不同干预场景下量化它们对最终推理结果的实际影响，从而实现缺失步骤的自动化添加和冗余步骤的修剪。在各种数学和常识推理基准测试上的广泛实验结果确认，在不牺牲准确性的情况下显著提高了推理效率并减少了标记使用量。我们的工作为提高LLM推理性能和成本效益提供了一个有希望的方向。 

---
# Dataset of News Articles with Provenance Metadata for Media Relevance Assessment 

**Title (ZH)**: 包含来源元数据的新闻文章数据集用于媒体相关性评估 

**Authors**: Tomas Peterka, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2506.09847)  

**Abstract**: Out-of-context and misattributed imagery is the leading form of media manipulation in today's misinformation and disinformation landscape. The existing methods attempting to detect this practice often only consider whether the semantics of the imagery corresponds to the text narrative, missing manipulation so long as the depicted objects or scenes somewhat correspond to the narrative at hand. To tackle this, we introduce News Media Provenance Dataset, a dataset of news articles with provenance-tagged images. We formulate two tasks on this dataset, location of origin relevance (LOR) and date and time of origin relevance (DTOR), and present baseline results on six large language models (LLMs). We identify that, while the zero-shot performance on LOR is promising, the performance on DTOR hinders, leaving room for specialized architectures and future work. 

**Abstract (ZH)**: 脱离上下文和误归因的图像已成为当今信息误导和虚假信息landscape中的主要媒体操纵形式。现有的检测方法往往仅考虑图像的语义是否与文本叙述相符，只要图像中描绘的对象或场景大致符合叙述，就会忽略操纵。为应对这一挑战，我们引入了新闻媒体溯源数据集，该数据集包含带有溯源标记的新闻文章。我们在该数据集上制定了两个任务：起始地点相关性（LOR）和起始时间相关性（DTOR），并在这六个大型语言模型（LLMs）上呈现了基线结果。我们发现，虽然LOR的零样本性能令人鼓舞，但DTOR的性能受限，为专门架构和未来工作留下了空间。 

---
# Learning to Align: Addressing Character Frequency Distribution Shifts in Handwritten Text Recognition 

**Title (ZH)**: 学习对齐：解决手写文本识别中的字符频率分布变化问题 

**Authors**: Panagiotis Kaliosis, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09846)  

**Abstract**: Handwritten text recognition aims to convert visual input into machine-readable text, and it remains challenging due to the evolving and context-dependent nature of handwriting. Character sets change over time, and character frequency distributions shift across historical periods or regions, often causing models trained on broad, heterogeneous corpora to underperform on specific subsets. To tackle this, we propose a novel loss function that incorporates the Wasserstein distance between the character frequency distribution of the predicted text and a target distribution empirically derived from training data. By penalizing divergence from expected distributions, our approach enhances both accuracy and robustness under temporal and contextual intra-dataset shifts. Furthermore, we demonstrate that character distribution alignment can also improve existing models at inference time without requiring retraining by integrating it as a scoring function in a guided decoding scheme. Experimental results across multiple datasets and architectures confirm the effectiveness of our method in boosting generalization and performance. We open source our code at this https URL. 

**Abstract (ZH)**: 基于手写文本识别旨在将视觉输入转换为机器可读文本，但由于手写笔迹随时间演化和依赖具体情境，这一任务仍然具有挑战性。字符集随时间变化，不同时期或地区字符频率分布发生变化，常常导致在特定子集上表现不佳。为此，我们提出了一种新的损失函数，该函数结合了预测文本中的字符频率分布与基于训练数据经验推导出的目标分布之间的 Wasserstein 距离。通过惩罚与期望分布的偏差，我们的方法在时间性和情境性内部数据集变化下提高了准确性和稳健性。此外，我们展示了字符分布对齐也可以在推理时提高现有模型的性能，而无需重新训练，通过将其整合为引导解码方案中的打分函数来实现。跨多个数据集和架构的实验结果证实了我们方法在增强泛化能力和性能方面的有效性。我们在 GitHub（此链接请替换为实际链接地址）开源了代码。 

---
# OctoNav: Towards Generalist Embodied Navigation 

**Title (ZH)**: OctoNav: 向 général 化自主导航迈进 

**Authors**: Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09839)  

**Abstract**: Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods. 

**Abstract (ZH)**: 嵌入式导航是广泛追求的嵌入式AI基础支柱。然而，之前的导航研究被划分为不同的任务/能力，例如ObjNav、ImgNav和VLN，它们在任务目标和模态上有所不同，导致数据集和方法独立设计。在此工作中，我们朝着通用导航代理迈进，这些代理能够遵循自由形式的指令，这些指令结合了多种模态和多种能力的任意复合。为此，我们提出了一种大规模基准及其相应的方法，称为OctoNav-Bench和OctoNav-R1。具体来说，OctoNav-Bench具有连续环境，并通过设计的注释流水线构建。我们精心策划指令-轨迹对，指令形式自由多样，包含任意模态和能力。此外，我们在OctoNav-Bench中构建了一个Think-Before-Action（TBA-CoT）数据集，提供动作背后的思维过程。对于OctoNav-R1，我们在MLLMs基础上构建并将其适应为VLA类型模型，可以仅基于2D视觉观察生成低级动作。同时，我们设计了一种混合训练范式（HTP），包括三个阶段：Action-/TBA-SFT、Nav-GPRO和在线强化学习阶段。每个阶段包含专门设计的学习策略和奖励。重要的是，对于TBA-SFT和Nav-GPRO设计，我们受OpenAI-o1和DeepSeek-R1的启发，通过思考后再作答展示了出色的推理能力。因此，我们旨在探索如何在嵌入式导航领域实现思考后再行动，以提高模型的推理能力。具体而言，我们提出了TBA-SFT利用TBA-CoT数据集对模型进行微调作为冷启动语句，然后利用Nav-GPRO提升其推理能力。最后，OctoNav-R1在性能上优于先前的方法。 

---
# DynaSplat: Dynamic-Static Gaussian Splatting with Hierarchical Motion Decomposition for Scene Reconstruction 

**Title (ZH)**: DynaSplat: 动静态高斯点云渲染及其层次运动分解场景重建 

**Authors**: Junli Deng, Ping Shi, Qipei Li, Jinyang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09836)  

**Abstract**: Reconstructing intricate, ever-changing environments remains a central ambition in computer vision, yet existing solutions often crumble before the complexity of real-world dynamics. We present DynaSplat, an approach that extends Gaussian Splatting to dynamic scenes by integrating dynamic-static separation and hierarchical motion modeling. First, we classify scene elements as static or dynamic through a novel fusion of deformation offset statistics and 2D motion flow consistency, refining our spatial representation to focus precisely where motion matters. We then introduce a hierarchical motion modeling strategy that captures both coarse global transformations and fine-grained local movements, enabling accurate handling of intricate, non-rigid motions. Finally, we integrate physically-based opacity estimation to ensure visually coherent reconstructions, even under challenging occlusions and perspective shifts. Extensive experiments on challenging datasets reveal that DynaSplat not only surpasses state-of-the-art alternatives in accuracy and realism but also provides a more intuitive, compact, and efficient route to dynamic scene reconstruction. 

**Abstract (ZH)**: 重构复杂多变的环境仍是计算机视觉领域的核心目标，现有解决方案往往难以应对真实世界动态的复杂性。我们提出了DynaSplat方法，该方法通过结合动态-静态分离和分层运动建模，将高斯点云扩展到动态场景。首先，我们通过一种新颖的形变偏移统计与二维运动流一致性融合方法，对场景元素进行静态或动态分类，从而精细化空间表示，集中在关键的运动区域。然后，我们引入了一种分层运动建模策略，既能捕捉粗粒度全局变换，又能捕捉细粒度局部运动，从而准确处理复杂的非刚性运动。最后，我们整合了基于物理的不透明度估计，确保在挑战性的遮挡和视角变化情况下，重构结果具有视觉一致性。广泛的实验结果表明，DynaSplat不仅在准确性和真实性上超越了现有最先进的方法，还提供了一条更为直观、紧凑且高效的动态场景重构途径。 

---
# EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection 

**Title (ZH)**: EmoNet-Voice：一种细粒度且专家验证的语音情感识别基准。 

**Authors**: Christoph Schuhmann, Robert Kaczmarczyk, Gollam Rabby, Felix Friedrich, Maurice Kraus, Kourosh Nadi, Huu Nguyen, Kristian Kersting, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2506.09827)  

**Abstract**: The advancement of text-to-speech and audio generation models necessitates robust benchmarks for evaluating the emotional understanding capabilities of AI systems. Current speech emotion recognition (SER) datasets often exhibit limitations in emotional granularity, privacy concerns, or reliance on acted portrayals. This paper introduces EmoNet-Voice, a new resource for speech emotion detection, which includes EmoNet-Voice Big, a large-scale pre-training dataset (featuring over 4,500 hours of speech across 11 voices, 40 emotions, and 4 languages), and EmoNet-Voice Bench, a novel benchmark dataset with human expert annotations. EmoNet-Voice is designed to evaluate SER models on a fine-grained spectrum of 40 emotion categories with different levels of intensities. Leveraging state-of-the-art voice generation, we curated synthetic audio snippets simulating actors portraying scenes designed to evoke specific emotions. Crucially, we conducted rigorous validation by psychology experts who assigned perceived intensity labels. This synthetic, privacy-preserving approach allows for the inclusion of sensitive emotional states often absent in existing datasets. Lastly, we introduce Empathic Insight Voice models that set a new standard in speech emotion recognition with high agreement with human experts. Our evaluations across the current model landscape exhibit valuable findings, such as high-arousal emotions like anger being much easier to detect than low-arousal states like concentration. 

**Abstract (ZH)**: 文本转语音和音频生成模型的进步需要稳健的基准来评估AI系统的情感理解能力。现有的语音情感识别（SER）数据集往往在情感精细度、隐私问题或依赖于表演呈现方面存在局限。本文介绍了一种新的语音情感检测资源EmoNet-Voice，其中包括EmoNet-Voice Big（一个大规模预训练数据集，包含超过4500小时的语音数据，涵盖11种声音、40种情感和4种语言）和EmoNet-Voice Bench（一个新型基准数据集，包含人类专家注释）。EmoNet-Voice旨在评估SER模型在40种不同情感强度级别的精细情感维度上的表现。借助最新的语音生成技术，我们精心制作了模拟演员表演特定情感场景的合成音频片段。至关重要的是，我们通过心理学专家的严格验证，分配了感知强度标签。这种合成且保护隐私的方法使得敏感的情感状态能够被包含在内，这些状态在现有数据集中往往缺失。最后，我们引入了Empathic Insight Voice模型，这些模型在语音情感识别方面达到了新的标准，与人类专家的判断高度一致。我们对当前模型景观的评估显示出有价值的结果，例如，高度激动的情感如愤怒比低激动状态如专注更容易被检测到。 

---
# Superstudent intelligence in thermodynamics 

**Title (ZH)**: 超学生在热力学中的智能 

**Authors**: Rebecca Loubet, Pascal Zittlau, Marco Hoffmann, Luisa Vollmer, Sophie Fellenz, Heike Leitte, Fabian Jirasek, Johannes Lenhard, Hans Hasse  

**Link**: [PDF](https://arxiv.org/pdf/2506.09822)  

**Abstract**: In this short note, we report and analyze a striking event: OpenAI's large language model o3 has outwitted all students in a university exam on thermodynamics. The thermodynamics exam is a difficult hurdle for most students, where they must show that they have mastered the fundamentals of this important topic. Consequently, the failure rates are very high, A-grades are rare - and they are considered proof of the students' exceptional intellectual abilities. This is because pattern learning does not help in the exam. The problems can only be solved by knowledgeably and creatively combining principles of thermodynamics. We have given our latest thermodynamics exam not only to the students but also to OpenAI's most powerful reasoning model, o3, and have assessed the answers of o3 exactly the same way as those of the students. In zero-shot mode, the model o3 solved all problems correctly, better than all students who took the exam; its overall score was in the range of the best scores we have seen in more than 10,000 similar exams since 1985. This is a turning point: machines now excel in complex tasks, usually taken as proof of human intellectual capabilities. We discuss the consequences this has for the work of engineers and the education of future engineers. 

**Abstract (ZH)**: 开放AI的大语言模型o3在热力学大学考试中击败所有学生：一项引人注目的事件及其影响 

---
# CoRT: Code-integrated Reasoning within Thinking 

**Title (ZH)**: CoRT: 代码集成推理 

**Authors**: Chengpeng Li, Zhengyang Tang, Ziniu Li, Mingfeng Xue, Keqin Bao, Tian Ding, Ruoyu Sun, Benyou Wang, Xiang Wang, Junyang Lin, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09820)  

**Abstract**: Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model's internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4\% and 8\% absolute improvements on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30\% fewer tokens for the 32B model and 50\% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at this https URL. 

**Abstract (ZH)**: CoRT：一种用于教学大型推理模型高效利用代码解释器的后训练框架 

---
# A theoretical framework for self-supervised contrastive learning for continuous dependent data 

**Title (ZH)**: 自监督对比学习的理论框架用于连续依赖数据 

**Authors**: Alexander Marusov, Alexander Yuhay, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.09785)  

**Abstract**: Self-supervised learning (SSL) has emerged as a powerful approach to learning representations, particularly in the field of computer vision. However, its application to dependent data, such as temporal and spatio-temporal domains, remains underexplored. Besides, traditional contrastive SSL methods often assume \emph{semantic independence between samples}, which does not hold for dependent data exhibiting complex correlations. We propose a novel theoretical framework for contrastive SSL tailored to \emph{continuous dependent data}, which allows the nearest samples to be semantically close to each other. In particular, we propose two possible \textit{ground truth similarity measures} between objects -- \emph{hard} and \emph{soft} closeness. Under it, we derive an analytical form for the \textit{estimated similarity matrix} that accommodates both types of closeness between samples, thereby introducing dependency-aware loss functions. We validate our approach, \emph{Dependent TS2Vec}, on temporal and spatio-temporal downstream problems. Given the dependency patterns presented in the data, our approach surpasses modern ones for dependent data, highlighting the effectiveness of our theoretically grounded loss functions for SSL in capturing spatio-temporal dependencies. Specifically, we outperform TS2Vec on the standard UEA and UCR benchmarks, with accuracy improvements of $4.17$\% and $2.08$\%, respectively. Furthermore, on the drought classification task, which involves complex spatio-temporal patterns, our method achieves a $7$\% higher ROC-AUC score. 

**Abstract (ZH)**: 自监督学习（SSL）已成为学习表示的强大方法，特别是在计算机视觉领域。然而，其在时间依赖性数据如时间域和空时域中的应用仍待进一步探索。此外，传统的对比自监督学习方法通常假设样本之间具有语义独立性，这对于展示复杂相关性的依赖性数据并不成立。我们提出了一种针对连续依赖性数据的新型对比自监督学习理论框架，该框架允许最近邻样本彼此具有语义上的接近性。特别地，我们提出了两种可能的“真实相似性度量”——“硬接近”和“软接近”。在此基础上，我们推导出一种能够适应样本之间两种接近性的估计相似性矩阵的形式，并由此引入了依赖性感知的损失函数。我们在时间序列和空时下游问题上验证了我们的方法，即Dependent TS2Vec。对于数据中展示的依赖性模式，我们的方法超越了现代依赖性数据方法，突显了我们理论支撑的损失函数在自监督学习中捕捉空时依赖性的有效性。具体而言，我们在标准UEA和UCR基准测试上分别取得了4.17%和2.08%的准确性提升。此外，在涉及复杂空时模式的干旱分类任务中，我们的方法获得了7%更高的ROC-AUC分数。 

---
# Q-SAM2: Accurate Quantization for Segment Anything Model 2 

**Title (ZH)**: Q-SAM2: 准确量化分割万物模型 

**Authors**: Nicola Farronato, Florian Scheidegger, Mattia Rigotti, Cristiano Malossi, Michele Magno, Haotong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09782)  

**Abstract**: The Segment Anything Model 2 (SAM2) has gained significant attention as a foundational approach for promptable image and video segmentation. However, its expensive computational and memory consumption poses a severe challenge for its application in resource-constrained scenarios. In this paper, we propose an accurate low-bit quantization method for efficient SAM2, termed Q-SAM2. To address the performance degradation caused by the singularities in weight and activation distributions during quantization, Q-SAM2 introduces two novel technical contributions. We first introduce a linear layer calibration method for low-bit initialization of SAM2, which minimizes the Frobenius norm over a small image batch to reposition weight distributions for improved quantization. We then propose a Quantization-Aware Training (QAT) pipeline that applies clipping to suppress outliers and allows the network to adapt to quantization thresholds during training. Our comprehensive experiments demonstrate that Q-SAM2 allows for highly accurate inference while substantially improving efficiency. Both quantitative and visual results show that our Q-SAM2 surpasses existing state-of-the-art general quantization schemes, especially for ultra-low 2-bit quantization. While designed for quantization-aware training, our proposed calibration technique also proves effective in post-training quantization, achieving up to a 66% mIoU accuracy improvement over non-calibrated models. 

**Abstract (ZH)**: Q-SAM2：一种高效的低比特量化方法 

---
# Inverting Black-Box Face Recognition Systems via Zero-Order Optimization in Eigenface Space 

**Title (ZH)**: 通过特征脸空间的零阶优化反向推理黑盒面部识别系统 

**Authors**: Anton Razzhigaev, Matvey Mikhalchuk, Klim Kireev, Igor Udovichenko, Andrey Kuznetsov, Aleksandr Petiushko  

**Link**: [PDF](https://arxiv.org/pdf/2506.09777)  

**Abstract**: Reconstructing facial images from black-box recognition models poses a significant privacy threat. While many methods require access to embeddings, we address the more challenging scenario of model inversion using only similarity scores. This paper introduces DarkerBB, a novel approach that reconstructs color faces by performing zero-order optimization within a PCA-derived eigenface space. Despite this highly limited information, experiments on LFW, AgeDB-30, and CFP-FP benchmarks demonstrate that DarkerBB achieves state-of-the-art verification accuracies in the similarity-only setting, with competitive query efficiency. 

**Abstract (ZH)**: 从黑色盒模型重建面部图像构成了重大的隐私威胁。尽管许多方法需要访问嵌入式信息，我们通过仅使用相似度分数来解决更具挑战性的模型反向工程问题。本文提出了DarkerBB，一种新颖的方法，通过在PCA衍生的特征脸空间中进行零阶优化来重建彩色面部。尽管仅限于此高度有限的信息，实验结果表明，在仅使用相似度分数的情况下，DarkerBB实现了最先进的验证准确率，并且具有竞争力的查询效率。 

---
# Load-Aware Training Scheduling for Model Circulation-based Decentralized Federated Learning 

**Title (ZH)**: 基于模型流通的去中心化联邦学习的负载感知训练调度 

**Authors**: Haruki Kainuma, Takayuki Nishio  

**Link**: [PDF](https://arxiv.org/pdf/2506.09769)  

**Abstract**: This paper proposes Load-aware Tram-FL, an extension of Tram-FL that introduces a training scheduling mechanism to minimize total training time in decentralized federated learning by accounting for both computational and communication loads. The scheduling problem is formulated as a global optimization task, which-though intractable in its original form-is made solvable by decomposing it into node-wise subproblems. To promote balanced data utilization under non-IID distributions, a variance constraint is introduced, while the overall training latency, including both computation and communication costs, is minimized through the objective function. Simulation results on MNIST and CIFAR-10 demonstrate that Load-aware Tram-FL significantly reduces training time and accelerates convergence compared to baseline methods. 

**Abstract (ZH)**: Load-aware Tram-FL：一种考虑计算和通信负载的卸载调度机制以减少去中心化联邦学习的总训练时间 

---
# Intelligent Design 4.0: Paradigm Evolution Toward the Agentic AI Era 

**Title (ZH)**: 智能设计4.0：面向代理人工智能时代的范式演进 

**Authors**: Shuo Jiang, Min Xie, Frank Youhua Chen, Jian Ma, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09755)  

**Abstract**: Research and practice in Intelligent Design (ID) have significantly enhanced engineering innovation, efficiency, quality, and productivity over recent decades, fundamentally reshaping how engineering designers think, behave, and interact with design processes. The recent emergence of Foundation Models (FMs), particularly Large Language Models (LLMs), has demonstrated general knowledge-based reasoning capabilities, and open new paths and avenues for further transformation in engineering design. In this context, this paper introduces Intelligent Design 4.0 (ID 4.0) as an emerging paradigm empowered by agentic AI systems. We review the historical evolution of ID across four distinct stages: rule-based expert systems, task-specific machine learning models, large-scale foundation AI models, and the recent emerging paradigm of multi-agent collaboration. We propose a conceptual framework for ID 4.0 and discuss its potential to support end-to-end automation of engineering design processes through coordinated, autonomous multi-agent-based systems. Furthermore, we discuss future perspectives to enhance and fully realize ID 4.0's potential, including more complex design scenarios, more practical design implementations, novel agent coordination mechanisms, and autonomous design goal-setting with better human value alignment. In sum, these insights lay a foundation for advancing Intelligent Design toward greater adaptivity, autonomy, and effectiveness in addressing increasingly complex design challenges. 

**Abstract (ZH)**: 智能设计4.0：基于自主人工智能系统的新兴范式研究与实践 

---
# Large Language Models for Design Structure Matrix Optimization 

**Title (ZH)**: 大型语言模型在设计结构矩阵优化中的应用 

**Authors**: Shuo Jiang, Min Xie, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09749)  

**Abstract**: In complex engineering systems, the interdependencies among components or development activities are often modeled and analyzed using Design Structure Matrix (DSM). Reorganizing elements within a DSM to minimize feedback loops and enhance modularity or process efficiency constitutes a challenging combinatorial optimization (CO) problem in engineering design and operations. As problem sizes increase and dependency networks become more intricate, traditional optimization methods that solely use mathematical heuristics often fail to capture the contextual nuances and struggle to deliver effective solutions. In this study, we explore the potential of Large Language Models (LLMs) for helping solve such CO problems by leveraging their capabilities for advanced reasoning and contextual understanding. We propose a novel LLM-based framework that integrates network topology with contextual domain knowledge for iterative optimization of DSM element sequencing - a common CO problem. Experiments on various DSM cases show that our method consistently achieves faster convergence and superior solution quality compared to both stochastic and deterministic baselines. Notably, we find that incorporating contextual domain knowledge significantly enhances optimization performance regardless of the chosen LLM backbone. These findings highlight the potential of LLMs to solve complex engineering CO problems by combining semantic and mathematical reasoning. This approach paves the way towards a new paradigm in LLM-based engineering design optimization. 

**Abstract (ZH)**: 利用大型语言模型解决工程系统中的组合优化问题：以设计结构矩阵元素序列为案例 

---
# Feature Engineering for Agents: An Adaptive Cognitive Architecture for Interpretable ML Monitoring 

**Title (ZH)**: 代理的特征工程：一种可解释的ML监控自适应认知架构 

**Authors**: Gusseppe Bravo-Rocca, Peini Liu, Jordi Guitart, Rodrigo M Carrillo-Larco, Ajay Dholakia, David Ellison  

**Link**: [PDF](https://arxiv.org/pdf/2506.09742)  

**Abstract**: Monitoring Machine Learning (ML) models in production environments is crucial, yet traditional approaches often yield verbose, low-interpretability outputs that hinder effective decision-making. We propose a cognitive architecture for ML monitoring that applies feature engineering principles to agents based on Large Language Models (LLMs), significantly enhancing the interpretability of monitoring outputs. Central to our approach is a Decision Procedure module that simulates feature engineering through three key steps: Refactor, Break Down, and Compile. The Refactor step improves data representation to better capture feature semantics, allowing the LLM to focus on salient aspects of the monitoring data while reducing noise and irrelevant information. Break Down decomposes complex information for detailed analysis, and Compile integrates sub-insights into clear, interpretable outputs. This process leads to a more deterministic planning approach, reducing dependence on LLM-generated planning, which can sometimes be inconsistent and overly general. The combination of feature engineering-driven planning and selective LLM utilization results in a robust decision support system, capable of providing highly interpretable and actionable insights. Experiments using multiple LLMs demonstrate the efficacy of our approach, achieving significantly higher accuracy compared to various baselines across several domains. 

**Abstract (ZH)**: 基于大型语言模型的特征工程原理的机器学习监控认知架构：提高监控输出的可解释性 

---
# ELBO-T2IAlign: A Generic ELBO-Based Method for Calibrating Pixel-level Text-Image Alignment in Diffusion Models 

**Title (ZH)**: 基于ELBO的通用方法：用于校准扩散模型中像素级文本-图像对齐的ELBO-T2IAlign 

**Authors**: Qin Zhou, Zhiyang Zhang, Jinglong Wang, Xiaobin Li, Jing Zhang, Qian Yu, Lu Sheng, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09740)  

**Abstract**: Diffusion models excel at image generation. Recent studies have shown that these models not only generate high-quality images but also encode text-image alignment information through attention maps or loss functions. This information is valuable for various downstream tasks, including segmentation, text-guided image editing, and compositional image generation. However, current methods heavily rely on the assumption of perfect text-image alignment in diffusion models, which is not the case. In this paper, we propose using zero-shot referring image segmentation as a proxy task to evaluate the pixel-level image and class-level text alignment of popular diffusion models. We conduct an in-depth analysis of pixel-text misalignment in diffusion models from the perspective of training data bias. We find that misalignment occurs in images with small sized, occluded, or rare object classes. Therefore, we propose ELBO-T2IAlign, a simple yet effective method to calibrate pixel-text alignment in diffusion models based on the evidence lower bound (ELBO) of likelihood. Our method is training-free and generic, eliminating the need to identify the specific cause of misalignment and works well across various diffusion model architectures. Extensive experiments on commonly used benchmark datasets on image segmentation and generation have verified the effectiveness of our proposed calibration approach. 

**Abstract (ZH)**: 扩散模型在图像生成中表现出色。最近的研究表明，这些模型不仅可以生成高质量的图像，还能通过注意力图或损失函数编码文本-图像对齐信息。这些信息对包括分割、文本引导的图像编辑和组成性图像生成在内的多种下游任务具有价值。然而，当前方法严重依赖于扩散模型中文本-图像对齐完美性的假设，实际情况并非如此。本文提出使用零-shot 引用图像分割作为代理任务，评估流行扩散模型的像素级图像对齐和类别级文本对齐。我们从训练数据偏差的角度深入分析了扩散模型中的像素-文本对齐偏差问题。我们发现，在小尺寸、被遮挡或罕见对象类别图像中存在对齐偏差。因此，我们提出了基于似然性证据下界（ELBO）的ELBO-T2IAlign方法，以简单有效的方式校准扩散模型中的像素-文本对齐。该方法无需训练，并具有通用性，无需识别对齐偏差的具体原因，适用于各种扩散模型架构。 extensive实验表明，我们的校准方法在常用的图像分割和生成基准数据集上具有有效性。 

---
# Vision Matters: Simple Visual Perturbations Can Boost Multimodal Math Reasoning 

**Title (ZH)**: 视觉很重要：简单的视觉扰动可以提升多模态数学推理能力 

**Authors**: Yuting Li, Lai Wei, Kaipeng Zheng, Jingyuan Huang, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09736)  

**Abstract**: Despite the rapid progress of multimodal large language models (MLLMs), they have largely overlooked the importance of visual processing. In a simple yet revealing experiment, we interestingly find that language-only models, when provided with image captions, can achieve comparable or even better performance than MLLMs that consume raw visual inputs. This suggests that current MLLMs may generate accurate visual descriptions but fail to effectively integrate them during reasoning. Motivated by this, we propose a simple visual perturbation framework that enhances perceptual robustness without requiring algorithmic modifications or additional training data. Our approach introduces three targeted perturbations: distractor concatenation, dominance-preserving mixup, and random rotation, that can be easily integrated into existing post-training pipelines including SFT, DPO, and GRPO. Through extensive experiments across multiple datasets, we demonstrate consistent improvements in mathematical reasoning performance, with gains comparable to those achieved through algorithmic changes. Additionally, we achieve competitive performance among open-source 7B RL-tuned models by training Qwen2.5-VL-7B with visual perturbation. Through comprehensive ablation studies, we analyze the effectiveness of different perturbation strategies, revealing that each perturbation type contributes uniquely to different aspects of visual reasoning. Our findings highlight the critical role of visual perturbation in multimodal mathematical reasoning: better reasoning begins with better seeing. Our code is available at this https URL. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）取得了快速进展，但它们在视觉处理的重要性上还缺乏足够的关注。在一项简单而富有启发性的实验中，我们发现，仅依靠语言的语言模型，在获得图像描述后，可以实现与消费原始视觉输入的MLLM相当甚至更优的性能。这表明当前的MLLM可能能够生成准确的视觉描述，但在推理过程中未能有效整合它们。受此启发，我们提出了一种简单的视觉扰动框架，该框架可以增强感知鲁棒性，而无需进行算法修改或额外的数据训练。我们的方法引入了三种有针对性的扰动：干扰项连接、保持主导性的混合、以及随机旋转，这些扰动可以轻松集成到现有的后训练管道中，包括SFT、DPO和GRPO。通过在多个数据集上的广泛实验，我们展示了在数学推理性能上的一致改进，这些改进与通过算法变化所达到的增益相当。此外，通过视觉扰动训练Qwen2.5-VL-7B，我们实现了开源7B RL调优模型的竞争力。通过全面的消融研究，我们分析了不同扰动策略的有效性，揭示了每种扰动类型对视觉推理的不同方面具有独特的贡献。我们的研究结果强调了在多模态数学推理中视觉扰动的关键作用：更好的推理始于更好的视觉能力。我们的代码可在以下链接获取：this https URL 

---
# AtmosMJ: Revisiting Gating Mechanism for AI Weather Forecasting Beyond the Year Scale 

**Title (ZH)**: AtmosMJ: 重新审视超越年尺度的AI天气预报中的门控机制 

**Authors**: Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2506.09733)  

**Abstract**: The advent of Large Weather Models (LWMs) has marked a turning point in data-driven forecasting, with many models now outperforming traditional numerical systems in the medium range. However, achieving stable, long-range autoregressive forecasts beyond a few weeks remains a significant challenge. Prevailing state-of-the-art models that achieve year-long stability, such as SFNO and DLWP-HPX, have relied on transforming input data onto non-standard spatial domains like spherical harmonics or HEALPix meshes. This has led to the prevailing assumption that such representations are necessary to enforce physical consistency and long-term stability. This paper challenges that assumption by investigating whether comparable long-range performance can be achieved on the standard latitude-longitude grid. We introduce AtmosMJ, a deep convolutional network that operates directly on ERA5 data without any spherical remapping. The model's stability is enabled by a novel Gated Residual Fusion (GRF) mechanism, which adaptively moderates feature updates to prevent error accumulation over long recursive simulations. Our results demonstrate that AtmosMJ produces stable and physically plausible forecasts for about 500 days. In quantitative evaluations, it achieves competitive 10-day forecast accuracy against models like Pangu-Weather and GraphCast, all while requiring a remarkably low training budget of 5.7 days on a V100 GPU. Our findings suggest that efficient architectural design, rather than non-standard data representation, can be the key to unlocking stable and computationally efficient long-range weather prediction. 

**Abstract (ZH)**: 大型天气模型的兴起标志着数据驱动预报的一个转折点，许多模型现在在中短期已经超越了传统的数值系统。然而，实现稳定且长期的自回归预报超过几周仍是一项重大挑战。现有的如SFNO和DLWP-HPX等领先的模型依靠将输入数据转换到非标准的空间域，如球谐函数或HEALPix网格，以确保物理一致性并实现长期稳定性。本文通过研究在标准经纬度网格上是否可以达到相似的长期性能来挑战这一假设。我们引入了AtmosMJ，这是一种直接在ERA5数据上运行的深度卷积网络，无需任何球面重构。通过一种新颖的门控残差融合（GRF）机制，该模型能够逐步更新特征以防止长时间递归模拟中的误差累积。我们的结果显示，AtmosMJ能够产生稳定且物理合理的预报长达约500天。在定性评估中，AtmosMJ在10天预报的准确性方面与Pangu-Weather和GraphCast等模型竞争，同时仅需一个V100 GPU训练5.7天的极低训练预算。我们的研究结果表明，有效的架构设计而非非标准数据表示可能是实现稳定且计算高效的长期天气预报的关键。 

---
# Non-Contact Health Monitoring During Daily Personal Care Routines 

**Title (ZH)**: 非接触式日常个人护理中的健康监测 

**Authors**: Xulin Ma, Jiankai Tang, Zhang Jiang, Songqin Cheng, Yuanchun Shi, Dong LI, Xin Liu, Daniel McDuff, Xiaojing Liu, Yuntao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09718)  

**Abstract**: Remote photoplethysmography (rPPG) enables non-contact, continuous monitoring of physiological signals and offers a practical alternative to traditional health sensing methods. Although rPPG is promising for daily health monitoring, its application in long-term personal care scenarios, such as mirror-facing routines in high-altitude environments, remains challenging due to ambient lighting variations, frequent occlusions from hand movements, and dynamic facial postures. To address these challenges, we present LADH (Long-term Altitude Daily Health), the first long-term rPPG dataset containing 240 synchronized RGB and infrared (IR) facial videos from 21 participants across five common personal care scenarios, along with ground-truth PPG, respiration, and blood oxygen signals. Our experiments demonstrate that combining RGB and IR video inputs improves the accuracy and robustness of non-contact physiological monitoring, achieving a mean absolute error (MAE) of 4.99 BPM in heart rate estimation. Furthermore, we find that multi-task learning enhances performance across multiple physiological indicators simultaneously. Dataset and code are open at this https URL. 

**Abstract (ZH)**: 长期内置高度日常健康监控的远程光体积描记术数据集（LADH） 

---
# TRIDENT: Temporally Restricted Inference via DFA-Enhanced Neural Traversal 

**Title (ZH)**: TRIDENT：基于DFA增强神经遍历的 temporally受限推理 

**Authors**: Vincenzo Collura, Karim Tit, Laura Bussi, Eleonora Giunchiglia, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.09701)  

**Abstract**: Large Language Models (LLMs) and other neural architectures have achieved impressive results across a variety of generative and classification tasks. However, they remain fundamentally ill-equipped to ensure that their outputs satisfy temporal constraints, such as those expressible in Linear Temporal Logic over finite traces (LTLf). In this paper, we introduce TRIDENT: a general and model-agnostic inference-time algorithm that guarantees compliance with such constraints without requiring any retraining. TRIDENT compiles LTLf formulas into a Deterministic Finite Automaton (DFA), which is used to guide a constrained variant of beam search. At each decoding step, transitions that would lead to constraint violations are masked, while remaining paths are dynamically re-ranked based on both the model's probabilities and the DFA's acceptance structure. We formally prove that the resulting sequences are guaranteed to satisfy the given LTLf constraints, and we empirically demonstrate that TRIDENT also improves output quality. We validate our approach on two distinct tasks: temporally constrained image-stream classification and controlled text generation. In both settings, TRIDENT achieves perfect constraint satisfaction, while comparison with the state of the art shows improved efficiency and high standard quality metrics. 

**Abstract (ZH)**: 大型语言模型(LLMs)和其他神经架构在生成和分类任务中取得了显著成果。然而，它们在确保其输出满足时间约束（如有限轨迹上的线性时态逻辑(LTLf)表达的时间约束）方面仍显得力不从心。本文介绍了TRIDENT：一个通用且模型无关的推理时算法，能够在无需重新训练的情况下保证输出满足此类约束。TRIDENT将LTLf公式编译为确定有限自动机(DFA)，用于指导受限变种的束搜索。在每次解码步骤中，会导致约束违反的过渡被掩藏，而剩余路径则基于模型概率和DFA接受结构动态重新排名。我们形式化证明了生成的序列保证满足给定的LTLf约束，并通过实验表明TRIDENT也提高了输出质量。我们分别在两个任务中验证了这种方法：时间受限的图像流分类和受控文本生成。在两种设置中，TRIDENT实现了完美的约束满足，与现有最佳方法相比，显示了更高的效率和高标准的质量指标。 

---
# Towards Practical Alzheimer's Disease Diagnosis: A Lightweight and Interpretable Spiking Neural Model 

**Title (ZH)**: 面向实际应用的阿尔茨海默病诊断：一种轻量级可解释的.spiiking神经网络模型 

**Authors**: Changwei Wu, Yifei Chen, Yuxin Du, Jinying Zong, Jie Dong, Mingxuan Liu, Yong Peng, Jin Fan, Feiwei Qin, Changmiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09695)  

**Abstract**: Early diagnosis of Alzheimer's Disease (AD), especially at the mild cognitive impairment (MCI) stage, is vital yet hindered by subjective assessments and the high cost of multimodal imaging modalities. Although deep learning methods offer automated alternatives, their energy inefficiency and computational demands limit real-world deployment, particularly in resource-constrained settings. As a brain-inspired paradigm, spiking neural networks (SNNs) are inherently well-suited for modeling the sparse, event-driven patterns of neural degeneration in AD, offering a promising foundation for interpretable and low-power medical diagnostics. However, existing SNNs often suffer from weak expressiveness and unstable training, which restrict their effectiveness in complex medical tasks. To address these limitations, we propose FasterSNN, a hybrid neural architecture that integrates biologically inspired LIF neurons with region-adaptive convolution and multi-scale spiking attention. This design enables sparse, efficient processing of 3D MRI while preserving diagnostic accuracy. Experiments on benchmark datasets demonstrate that FasterSNN achieves competitive performance with substantially improved efficiency and stability, supporting its potential for practical AD screening. Our source code is available at this https URL. 

**Abstract (ZH)**: 基于尖峰神经网络的阿尔茨海默病早期诊断：FasterSNN架构的研究 

---
# Reasoning Models Are More Easily Gaslighted Than You Think 

**Title (ZH)**: 推理模型比你想象的更容易受到 Gaslighting 

**Authors**: Bin Zhu, Hailong Yin, Jingjing Chen, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09677)  

**Abstract**: Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand misleading user input remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation prompts, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation prompt. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings reveal fundamental limitations in the robustness of reasoning models, highlighting the gap between step-by-step reasoning and belief persistence. 

**Abstract (ZH)**: Recent Advances in Reasoning-Centric Models: Evaluating their Robustness to Gaslighting Negation Prompts and Introducing GaslightingBench-R 

---
# Is Fine-Tuning an Effective Solution? Reassessing Knowledge Editing for Unstructured Data 

**Title (ZH)**: 微调是一个有效的解决方案吗？重新评估知识编辑对无结构数据的有效性 

**Authors**: Hao Xiong, Chuanyuan Tan, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09672)  

**Abstract**: Unstructured Knowledge Editing (UKE) is crucial for updating the relevant knowledge of large language models (LLMs). It focuses on unstructured inputs, such as long or free-form texts, which are common forms of real-world knowledge. Although previous studies have proposed effective methods and tested them, some issues exist: (1) Lack of Locality evaluation for UKE, and (2) Abnormal failure of fine-tuning (FT) based methods for UKE. To address these issues, we first construct two datasets, UnKEBench-Loc and AKEW-Loc (CF), by extending two existing UKE datasets with locality test data from the unstructured and structured views. This enables a systematic evaluation of the Locality of post-edited models. Furthermore, we identify four factors that may affect the performance of FT-based methods. Based on these factors, we conduct experiments to determine how the well-performing FT-based methods should be trained for the UKE task, providing a training recipe for future research. Our experimental results indicate that the FT-based method with the optimal setting (FT-UKE) is surprisingly strong, outperforming the existing state-of-the-art (SOTA). In batch editing scenarios, FT-UKE shows strong performance as well, with its advantage over SOTA methods increasing as the batch size grows, expanding the average metric lead from +6.78% to +10.80% 

**Abstract (ZH)**: 无结构知识编辑（UKE）是大型语言模型（LLMs）更新相关知识的关键。它专注于无结构输入，如长文本或自由格式文本，这些都是现实世界知识的常见形式。尽管先前的研究提出了一些有效的方法并进行了测试，但仍存在一些问题：(1) 缺乏无结构知识编辑的局部性评估，(2) 以微调（FT）为基础的方法在无结构知识编辑中出现异常失败。为了解决这些问题，我们首先通过扩展两个现有的UKE数据集并加入无结构和结构视角的局部性测试数据，构建了两个新的数据集UnKEBench-Loc和AKEW-Loc（CF），从而系统性地评估后编辑模型的局部性。此外，我们识别出了可能影响基于微调方法性能的四个因素，并基于这些因素进行了实验，以确定这些表现良好的基于微调的方法应如何为UKE任务进行训练，提供未来研究的训练方案。实验结果表明，具有最佳设置的基于微调的方法（FT-UKE）表现异常强大，优于现有最佳方法（SOTA）。在批量编辑场景中，FT-UKE同样表现出色，随着批次大小的增加，其相对于SOTA方法的优势增加，平均度量值领先优势从+6.78%扩大到+10.80%。 

---
# Empirical Quantification of Spurious Correlations in Malware Detection 

**Title (ZH)**: 恶意软件检测中虚假相关性的经验量化 

**Authors**: Bianca Perasso, Ludovico Lozza, Andrea Ponte, Luca Demetrio, Luca Oneto, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2506.09662)  

**Abstract**: End-to-end deep learning exhibits unmatched performance for detecting malware, but such an achievement is reached by exploiting spurious correlations -- features with high relevance at inference time, but known to be useless through domain knowledge. While previous work highlighted that deep networks mainly focus on metadata, none investigated the phenomenon further, without quantifying their impact on the decision. In this work, we deepen our understanding of how spurious correlation affects deep learning for malware detection by highlighting how much models rely on empty spaces left by the compiler, which diminishes the relevance of the compiled code. Through our seminal analysis on a small-scale balanced dataset, we introduce a ranking of two end-to-end models to better understand which is more suitable to be put in production. 

**Abstract (ZH)**: 端到端深度学习在检测恶意软件方面表现出色，但这种成就依赖于错误的相关性——在推理时具有高相关性的特征，但通过领域知识已知是无用的。尽管先前的工作指出深度网络主要关注元数据，但没有进一步探讨其对决策的影响。在本研究中，我们通过强调模型如何依赖编译器留下的空白空间，深化了对错误相关性如何影响恶意软件检测深度学习的理解，这些空白空间降低了编译代码的相关性。通过对小型平衡数据集的开创性分析，我们引入了两个端到端模型的排序，以更好地理解哪个模型更适合投入生产。 

---
# DGAE: Diffusion-Guided Autoencoder for Efficient Latent Representation Learning 

**Title (ZH)**: DGAE：扩散引导自编码器在高效潜在表示学习中的应用 

**Authors**: Dongxu Liu, Yuang Peng, Haomiao Tang, Yuwei Chen, Chunrui Han, Zheng Ge, Daxin Jiang, Mingxue Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09644)  

**Abstract**: Autoencoders empower state-of-the-art image and video generative models by compressing pixels into a latent space through visual tokenization. Although recent advances have alleviated the performance degradation of autoencoders under high compression ratios, addressing the training instability caused by GAN remains an open challenge. While improving spatial compression, we also aim to minimize the latent space dimensionality, enabling more efficient and compact representations. To tackle these challenges, we focus on improving the decoder's expressiveness. Concretely, we propose DGAE, which employs a diffusion model to guide the decoder in recovering informative signals that are not fully decoded from the latent representation. With this design, DGAE effectively mitigates the performance degradation under high spatial compression rates. At the same time, DGAE achieves state-of-the-art performance with a 2x smaller latent space. When integrated with Diffusion Models, DGAE demonstrates competitive performance on image generation for ImageNet-1K and shows that this compact latent representation facilitates faster convergence of the diffusion model. 

**Abstract (ZH)**: 自动编码器通过视觉标记化将像素压缩到潜在空间，增强最先进的图像和视频生成模型。尽管近期进展缓解了在高压缩比下自动编码器性能下降的问题，GAN引发的训练稳定性问题仍是一个开放挑战。在提高空间压缩的同时，我们还致力于最小化潜在空间维度，以实现更高效和紧凑的表示。为应对这些挑战，我们专注于提升解码器的表达能力。具体而言，我们提出了一种基于扩散模型的DGAE方法，通过扩散模型引导解码器恢复从潜在表示中未完全解码的有意义信号。通过这种设计，DGAE在高空间压缩率下有效地缓解了性能下降的问题。同时，DGAE以2倍较小的潜在空间实现了最先进的性能。将DGAE与扩散模型结合使用时，它在ImageNet-1K图像生成任务上显示出竞争力，并表明这种紧凑的潜在表示有助于扩散模型更快收敛。 

---
# HSENet: Hybrid Spatial Encoding Network for 3D Medical Vision-Language Understanding 

**Title (ZH)**: HSENet：用于3D医疗视觉-语言理解的混合空间编码网络 

**Authors**: Yanzhao Shi, Xiaodan Zhang, Junzhong Ji, Haoning Jiang, Chengxin Zheng, Yinong Wang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09634)  

**Abstract**: Automated 3D CT diagnosis empowers clinicians to make timely, evidence-based decisions by enhancing diagnostic accuracy and workflow efficiency. While multimodal large language models (MLLMs) exhibit promising performance in visual-language understanding, existing methods mainly focus on 2D medical images, which fundamentally limits their ability to capture complex 3D anatomical structures. This limitation often leads to misinterpretation of subtle pathologies and causes diagnostic hallucinations. In this paper, we present Hybrid Spatial Encoding Network (HSENet), a framework that exploits enriched 3D medical visual cues by effective visual perception and projection for accurate and robust vision-language understanding. Specifically, HSENet employs dual-3D vision encoders to perceive both global volumetric contexts and fine-grained anatomical details, which are pre-trained by dual-stage alignment with diagnostic reports. Furthermore, we propose Spatial Packer, an efficient multimodal projector that condenses high-resolution 3D spatial regions into a compact set of informative visual tokens via centroid-based compression. By assigning spatial packers with dual-3D vision encoders, HSENet can seamlessly perceive and transfer hybrid visual representations to LLM's semantic space, facilitating accurate diagnostic text generation. Experimental results demonstrate that our method achieves state-of-the-art performance in 3D language-visual retrieval (39.85% of R@100, +5.96% gain), 3D medical report generation (24.01% of BLEU-4, +8.01% gain), and 3D visual question answering (73.60% of Major Class Accuracy, +1.99% gain), confirming its effectiveness. Our code is available at this https URL. 

**Abstract (ZH)**: 自动化3D CT诊断赋能临床医生通过增强诊断准确性和工作流程效率及时做出基于证据的决策。现有方法主要集中在2D医学图像上，这从根本上限制了其捕捉复杂3D解剖结构的能力。受限于此，往往会误解细微的病理特征，导致诊断幻觉。本文提出了结合空间编码网络（HSENet），该框架通过有效的视觉感知和投影利用丰富的3D医学视觉线索，以实现准确和鲁棒的视觉语言理解。具体而言，HSENet 使用双3D视觉编码器感知全局体素上下文和精细的解剖细节，这些细节通过双重阶段对齐预训练诊断报告。此外，我们提出了空间打包器（Spatial Packer），这是一种高效多模态投影器，通过基于质心的压缩将高分辨率3D空间区域凝缩为一组信息丰富的视觉标记。通过赋予空间打包器双3D视觉编码器，HSENet 可以无缝地感知和传递混合视觉表示到大语言模型（LLM）的语义空间，促进准确的诊断文本生成。实验结果证明，我们的方法在3D语言视觉检索（39.85%的R@100，+5.96%的提高）、3D医学报告生成（24.01%的BLEU-4，+8.01%的提高）和3D视觉问答（73.60%的主要类别准确度，+1.99%的提高）中达到了最先进的性能，证实了其有效性。我们的代码可在以下链接获取。 

---
# Effective Red-Teaming of Policy-Adherent Agents 

**Title (ZH)**: 有效检验政策遵从代理 

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor  

**Link**: [PDF](https://arxiv.org/pdf/2506.09600)  

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks 

**Abstract (ZH)**: 面向任务的基于LLM的代理在严格政策领域中的应用及其安全挑战：一个针对恶意用户行为的威胁模型与防御策略研究 

---
# From Symbolic to Neural and Back: Exploring Knowledge Graph-Large Language Model Synergies 

**Title (ZH)**: 从符号到神经网络再回到符号：探索知识图谱-大规模语言模型协同效应 

**Authors**: Blaž Škrlj, Boshko Koloski, Senja Pollak, Nada Lavrač  

**Link**: [PDF](https://arxiv.org/pdf/2506.09566)  

**Abstract**: Integrating structured knowledge from Knowledge Graphs (KGs) into Large Language Models (LLMs) enhances factual grounding and reasoning capabilities. This survey paper systematically examines the synergy between KGs and LLMs, categorizing existing approaches into two main groups: KG-enhanced LLMs, which improve reasoning, reduce hallucinations, and enable complex question answering; and LLM-augmented KGs, which facilitate KG construction, completion, and querying. Through comprehensive analysis, we identify critical gaps and highlight the mutual benefits of structured knowledge integration. Compared to existing surveys, our study uniquely emphasizes scalability, computational efficiency, and data quality. Finally, we propose future research directions, including neuro-symbolic integration, dynamic KG updating, data reliability, and ethical considerations, paving the way for intelligent systems capable of managing more complex real-world knowledge tasks. 

**Abstract (ZH)**: 将知识图谱（KGs）结构化知识集成到大型语言模型（LLMs）中增强了事实基础和推理能力。本文综述性地探讨了KGs与LLMs之间的协同作用，将现有方法分类为两类：KG增强的LLMs，改善推理、减少幻觉并实现复杂问题回答；以及LLMs增强的KGs，促进KG构建、补充和完善以及查询。通过综合分析，我们指出了结构化知识集成的关键空白，并突显了这种互惠互利。与现有综述相比，我们的研究独特地强调了可扩展性、计算效率和数据质量。最后，我们提出了未来研究方向，包括神经符号集成、动态KG更新、数据可靠性及伦理考量，为进一步处理更复杂的现实世界知识任务铺平了道路。 

---
# AD^2-Bench: A Hierarchical CoT Benchmark for MLLM in Autonomous Driving under Adverse Conditions 

**Title (ZH)**: AD^2-Bench：在恶劣条件下的自主驾驶中面向MLLM的层级式链式推理基准 

**Authors**: Zhaoyang Wei, Chenhui Qiang, Bowen Jiang, Xumeng Han, Xuehui Yu, Zhenjun Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.09557)  

**Abstract**: Chain-of-Thought (CoT) reasoning has emerged as a powerful approach to enhance the structured, multi-step decision-making capabilities of Multi-Modal Large Models (MLLMs), is particularly crucial for autonomous driving with adverse weather conditions and complex traffic environments. However, existing benchmarks have largely overlooked the need for rigorous evaluation of CoT processes in these specific and challenging scenarios. To address this critical gap, we introduce AD^2-Bench, the first Chain-of-Thought benchmark specifically designed for autonomous driving with adverse weather and complex scenes. AD^2-Bench is meticulously constructed to fulfill three key criteria: comprehensive data coverage across diverse adverse environments, fine-grained annotations that support multi-step reasoning, and a dedicated evaluation framework tailored for assessing CoT performance. The core contribution of AD^2-Bench is its extensive collection of over 5.4k high-quality, manually annotated CoT instances. Each intermediate reasoning step in these annotations is treated as an atomic unit with explicit ground truth, enabling unprecedented fine-grained analysis of MLLMs' inferential processes under text-level, point-level, and region-level visual prompts. Our comprehensive evaluation of state-of-the-art MLLMs on AD^2-Bench reveals accuracy below 60%, highlighting the benchmark's difficulty and the need to advance robust, interpretable end-to-end autonomous driving systems. AD^2-Bench thus provides a standardized evaluation platform, driving research forward by improving MLLMs' reasoning in autonomous driving, making it an invaluable resource. 

**Abstract (ZH)**: AD^2-Bench：一种专门针对恶劣天气和复杂场景的自主驾驶链式推理基准 

---
# Tightly-Coupled LiDAR-IMU-Leg Odometry with Online Learned Leg Kinematics Incorporating Foot Tactile Information 

**Title (ZH)**: 紧耦合LiDAR-IMU-腿部里程计融合，结合足部触觉信息的在线学习腿部运动学 

**Authors**: Taku Okawara, Kenji Koide, Aoki Takanose, Shuji Oishi, Masashi Yokozuka, Kentaro Uno, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2506.09548)  

**Abstract**: In this letter, we present tightly coupled LiDAR-IMU-leg odometry, which is robust to challenging conditions such as featureless environments and deformable terrains. We developed an online learning-based leg kinematics model named the neural leg kinematics model, which incorporates tactile information (foot reaction force) to implicitly express the nonlinear dynamics between robot feet and the ground. Online training of this model enhances its adaptability to weight load changes of a robot (e.g., assuming delivery or transportation tasks) and terrain conditions. According to the \textit{neural adaptive leg odometry factor} and online uncertainty estimation of the leg kinematics model-based motion predictions, we jointly solve online training of this kinematics model and odometry estimation on a unified factor graph to retain the consistency of both. The proposed method was verified through real experiments using a quadruped robot in two challenging situations: 1) a sandy beach, representing an extremely featureless area with a deformable terrain, and 2) a campus, including multiple featureless areas and terrain types of asphalt, gravel (deformable terrain), and grass. Experimental results showed that our odometry estimation incorporating the \textit{neural leg kinematics model} outperforms state-of-the-art works. Our project page is available for further details: this https URL 

**Abstract (ZH)**: 本文介绍了紧耦合LiDAR-IMU腿部里程计，该方法在无特征环境和可变形地形等挑战条件下表现出高度鲁棒性。文中开发了一种基于在线学习的腿动力学模型——神经腿动力学模型，该模型整合了触觉信息（足反应力）以隐式表达机器人足部与地面之间的非线性动力学。该模型的在线训练增强了其对机器人载荷变化（例如执行递送或运输任务）和地形条件的适应性。根据神经自适应腿部里程计因子和基于腿动力学模型的运动预测的在线不确定性估计，文中在统一因子图上联合解决了该动力学模型的在线训练和里程计估计，以保持两者的一致性。通过在四足机器人上进行的实际试验验证了本方法，试验环境包括两个具有挑战性的场景：1）一个沙滩，代表一个极度无特征的可变形地形区域；2）一个校园，包括多个无特征区域和不同类型的地形，如沥青、碎石（可变形地形）和草地。实验结果表明，结合神经腿动力学模型的里程计估计优于现有最先进的方法。我们的项目页面可提供更多详情：this https URL。 

---
# Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models 

**Title (ZH)**: Athena: 以数据高效的过程奖励模型增强多模态推理 

**Authors**: Shuai Wang, Zhenhua Liu, Jiaheng Wei, Xuanwu Yin, Dong Li, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2506.09532)  

**Abstract**: We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks. 

**Abstract (ZH)**: Athena-PRM: 多模态过程奖励模型及其在复杂推理问题评估中的应用 

---
# Neural Functions for Learning Periodic Signal 

**Title (ZH)**: 神经网络在学习周期信号中的功能 

**Authors**: Woojin Cho, Minju Jo, Kookjin Lee, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.09526)  

**Abstract**: As function approximators, deep neural networks have served as an effective tool to represent various signal types. Recent approaches utilize multi-layer perceptrons (MLPs) to learn a nonlinear mapping from a coordinate to its corresponding signal, facilitating the learning of continuous neural representations from discrete data points. Despite notable successes in learning diverse signal types, coordinate-based MLPs often face issues of overfitting and limited generalizability beyond the training region, resulting in subpar extrapolation performance. This study addresses scenarios where the underlying true signals exhibit periodic properties, either spatially or temporally. We propose a novel network architecture, which extracts periodic patterns from measurements and leverages this information to represent the signal, thereby enhancing generalization and improving extrapolation performance. We demonstrate the efficacy of the proposed method through comprehensive experiments, including the learning of the periodic solutions for differential equations, and time series imputation (interpolation) and forecasting (extrapolation) on real-world datasets. 

**Abstract (ZH)**: 基于坐标的多层感知机在表示具有周期性特性的信号中的应用：一种新型网络架构的提出与验证 

---
# Revisit What You See: Disclose Language Prior in Vision Tokens for Efficient Guided Decoding of LVLMs 

**Title (ZH)**: 重访你所看见的：在视觉标记中披露语言先验以实现高效的LVLM引导解码 

**Authors**: Beomsik Cho, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09522)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across various multimodal tasks by integrating visual perception with language understanding. However, conventional decoding strategies of LVLMs often fail to successfully utilize visual information, leading to visually ungrounded responses. While various approaches have been proposed to address this limitation, they typically require additional training, multi-step inference procedures, or external model dependencies. This paper introduces ReVisiT, a simple yet effective decoding method that references vision tokens to guide the text generation process in LVLMs. Our approach leverages the semantic information embedded within vision tokens by projecting them into the text token distribution space, and dynamically selecting the most relevant vision token at each decoding step through constrained divergence minimization. This selected vision token is then used to refine the output distribution to better incorporate visual semantics. Experiments on three LVLM hallucination benchmarks with two recent LVLMs demonstrate that ReVisiT consistently enhances visual grounding with minimal computational overhead. Moreover, our method achieves competitive or superior results relative to state-of-the-art baselines while reducing computational costs for up to $2\times$. 

**Abstract (ZH)**: 大规模 vision-language 模型 (LVLMs) 在多模态任务中通过结合视觉感知和语言理解展现了卓越的表现。然而，LVLMs 传统的解码策略往往无法有效利用视觉信息，导致生成的回答与视觉内容脱节。尽管提出了多种方法来解决这一限制，它们通常需要额外的训练、多步推理流程或外部模型依赖。本文引入了 ReVisiT，这是一种简单而有效的解码方法，通过引用视觉标记来指导 LVLMs 的文本生成过程。我们的方法通过将视觉标记投影到文本标记分布空间并动态选择与当前解码步骤最相关的视觉标记来利用嵌入在其内的语义信息，从而通过受限的偏差最小化进行选择。选择的视觉标记随后用于细化输出分布，使其更好地融入视觉语义。在两个最新 LVLMs 上对三个 LVLM 幻觉基准的实验表明，ReVisiT 以最小的计算开销一致地提升了视觉定位能力。此外，与最先进的基线方法相比，我们的方法在某些情况下实现了可竞争或更优的结果，同时将计算成本降低了最多 $2\times$。 

---
# How attention simplifies mental representations for planning 

**Title (ZH)**: 注意力如何简化 planning 中的心理表征 

**Authors**: Jason da Silva Castanheira, Nicholas Shea, Stephen M. Fleming  

**Link**: [PDF](https://arxiv.org/pdf/2506.09520)  

**Abstract**: Human planning is efficient -- it frugally deploys limited cognitive resources to accomplish difficult tasks -- and flexible -- adapting to novel problems and environments. Computational approaches suggest that people construct simplified mental representations of their environment, balancing the complexity of a task representation with its utility. These models imply a nested optimisation in which planning shapes perception, and perception shapes planning -- but the perceptual and attentional mechanisms governing how this interaction unfolds remain unknown. Here, we harness virtual maze navigation to characterise how spatial attention controls which aspects of a task representation enter subjective awareness and are available for planning. We find that spatial proximity governs which aspects of a maze are available for planning, and that when task-relevant information follows natural (lateralised) contours of attention, people can more easily construct simplified and useful maze representations. This influence of attention varies considerably across individuals, explaining differences in people's task representations and behaviour. Inspired by the 'spotlight of attention' analogy, we incorporate the effects of visuospatial attention into existing computational accounts of value-guided construal. Together, our work bridges computational perspectives on perception and decision-making to better understand how individuals represent their environments in aid of planning. 

**Abstract (ZH)**: 人类规划既高效又灵活——它精打细算地使用有限的认知资源来完成复杂的任务，并且能够适应新的问题和环境。计算方法表明人们构建了环境的简化心理表征，平衡任务表征的复杂性和实用性。这些模型暗示了一种嵌套的优化过程，在此过程中规划影响感知，而感知又影响规划——但调控这一交互过程的感知和注意力机制仍然未知。在这里，我们利用虚拟迷宫导航来研究空间注意力如何控制哪些任务表征进入主观意识并可供规划使用。我们发现，空间接近度决定了哪些迷宫方面可供规划使用，当与任务相关信息遵循自然（侧向化）的注意力轮廓时，人们更容易构建简化且有用的迷宫表征。这种注意力的影响在个体间差异很大，解释了人们之间任务表征和行为的差异。借鉴“注意力的探照灯”类比，我们将注意力的空间视觉效应纳入现有价值导向构念的计算解释中。我们的研究将关于感知和决策的计算视角结合起来，以更好地理解个体如何代表其环境以助于规划。 

---
# ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning 

**Title (ZH)**: ReasonMed: 一个包含37万多个agents生成的数据集，用于推动医疗推理研究 

**Authors**: Yu Sun, Xingyu Qian, Weiwen Xu, Hao Zhang, Chenghao Xiao, Long Li, Yu Rong, Wenbing Huang, Qifeng Bai, Tingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09513)  

**Abstract**: Though reasoning-based large language models (LLMs) have excelled in mathematics and programming, their capabilities in knowledge-intensive medical question answering remain underexplored. To address this, we introduce ReasonMed, the largest medical reasoning dataset, comprising 370k high-quality examples distilled from 1.7 million initial reasoning paths generated by various LLMs. ReasonMed is constructed through a \textit{multi-agent verification and refinement process}, where we design an \textit{Error Refiner} to enhance the reasoning paths by identifying and correcting error-prone steps flagged by a verifier. Leveraging ReasonMed, we systematically investigate best practices for training medical reasoning models and find that combining detailed Chain-of-Thought (CoT) reasoning with concise answer summaries yields the most effective fine-tuning strategy. Based on this strategy, we train ReasonMed-7B, which sets a new benchmark for sub-10B models, outperforming the prior best by 4.17\% and even exceeding LLaMA3.1-70B on PubMedQA by 4.60\%. 

**Abstract (ZH)**: 尽管基于推理的大语言模型（LLMs）在数学和编程方面表现出色，但它们在知识密集型医疗问题解答方面的能力仍较少被探索。为了解决这一问题，我们介绍了ReasonMed，这是目前最大的医疗推理数据集，包含370,000个高质量的例子，这些例子是从各种LLM生成的170万条初始推理路径中提炼出来的。ReasonMed通过一个“多智能体验证和精炼过程”构建起来，在这个过程中，我们设计了一个“错误精炼器”，通过识别并修正验证器标记的错误步骤，来增强推理路径。利用ReasonMed，我们系统地研究了训练医疗推理模型的最佳实践，并发现结合详细的思维链（CoT）推理与简洁的答案总结是最有效的微调策略。基于这种策略，我们训练了ReasonMed-7B，它为小于10B的模型设定了一个新的基准，相对于之前的最好成绩提升了4.17%，甚至在PubMedQA上超过了LLaMA3.1-70B 4.60%。 

---
# Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design 

**Title (ZH)**: 基于偏好高效强化学习：随机探索遇上实验设计 

**Authors**: Andreas Schlaginhaufen, Reda Ouhamma, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.09508)  

**Abstract**: We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries. 

**Abstract (ZH)**: 我们研究一般马尔可夫决策过程中的基于人工反馈的强化学习，其中代理通过轨迹级偏好比较进行学习。在这种设置中，一个核心挑战是设计算法以选择有信息量的偏好查询来识别潜在的奖励，并同时保证理论上的保证。我们提出了一种基于随机探索的元算法，该算法避免了乐观方法相关的计算挑战，并且具有可处理性。我们在温和的强化学习oracle假设下建立了遗憾和最后迭代的保证。为了改进查询复杂度，我们引入并分析了一种改进的算法，该算法收集轨迹对批次，并使用最优实验设计选择有信息量的比较查询。批结构还使得偏好查询的并行化成为可能，在实际部署中，反馈可以同时收集。实验评估证实，所提出的方法在需要少量偏好查询的情况下与基于奖励的强化学习方法具有竞争力。 

---
# TransXSSM: A Hybrid Transformer State Space Model with Unified Rotary Position Embedding 

**Title (ZH)**: 跨XSSM：一种结合统一旋转位置嵌入的混合变压器状态空间模型 

**Authors**: Bingheng Wu, Jingze Shi, Yifan Wu, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09507)  

**Abstract**: Transformers exhibit proficiency in capturing long-range dependencies, whereas State Space Models (SSMs) facilitate linear-time sequence modeling. Notwithstanding their synergistic potential, the integration of these architectures presents a significant challenge, primarily attributable to a fundamental incongruity in their respective positional encoding mechanisms: Transformers rely on explicit Rotary Position Embeddings (RoPE), while SSMs leverage implicit positional representations via convolutions. This divergence often precipitates discontinuities and suboptimal performance. To address this impediment, we propose a unified rotary position embedding (\textbf{\ourRoPE}) methodology, thereby establishing a consistent positional encoding framework for both self-attention and state-space components. Using this \ourRoPE, we introduce \textbf{\model}, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme. At a 4K sequence length, \model exhibits training and inference speeds that are \textbf{42.3\% and 29.5\% faster}, respectively, relative to standard Transformer models. It also delivers higher accuracy: under comparable settings, it surpasses a Transformer baseline by over 4\% on language modeling benchmarks. \model furthermore scales more effectively: \model-1.3B gains \textbf{7.22\%} in average accuracy over its 320M version (versus about 6\% gains for equivalent Transformers or SSMs). Our results show that unified positional encoding resolves positional incompatibility in hybrid models, enabling efficient, high-performance long-context modeling. 

**Abstract (ZH)**: 统一旋转位置嵌入的变压器-状态空间混合模型 

---
# A Unified Theory of Compositionality, Modularity, and Interpretability in Markov Decision Processes 

**Title (ZH)**: 马尔可夫决策过程中的组成性、模块化和可解释性统一理论 

**Authors**: Thomas J. Ringstrom, Paul R. Schrater  

**Link**: [PDF](https://arxiv.org/pdf/2506.09499)  

**Abstract**: We introduce Option Kernel Bellman Equations (OKBEs) for a new reward-free Markov Decision Process. Rather than a value function, OKBEs directly construct and optimize a predictive map called a state-time option kernel (STOK) to maximize the probability of completing a goal while avoiding constraint violations. STOKs are compositional, modular, and interpretable initiation-to-termination transition kernels for policies in the Options Framework of Reinforcement Learning. This means: 1) STOKs can be composed using Chapman-Kolmogorov equations to make spatiotemporal predictions for multiple policies over long horizons, 2) high-dimensional STOKs can be represented and computed efficiently in a factorized and reconfigurable form, and 3) STOKs record the probabilities of semantically interpretable goal-success and constraint-violation events, needed for formal verification. Given a high-dimensional state-transition model for an intractable planning problem, we can decompose it with local STOKs and goal-conditioned policies that are aggregated into a factorized goal kernel, making it possible to forward-plan at the level of goals in high-dimensions to solve the problem. These properties lead to highly flexible agents that can rapidly synthesize meta-policies, reuse planning representations across many tasks, and justify goals using empowerment, an intrinsic motivation function. We argue that reward-maximization is in conflict with the properties of compositionality, modularity, and interpretability. Alternatively, OKBEs facilitate these properties to support verifiable long-horizon planning and intrinsic motivation that scales to dynamic high-dimensional world-models. 

**Abstract (ZH)**: 基于选项内核贝尔曼方程的无奖励马尔可夫决策过程 

---
# EnerBridge-DPO: Energy-Guided Protein Inverse Folding with Markov Bridges and Direct Preference Optimization 

**Title (ZH)**: EnerBridge-DPO: 能量导向的蛋白质逆折叠方法结合马尔可夫桥和直接偏好优化 

**Authors**: Dingyi Rong, Haotian Lu, Wenzhuo Zheng, Fan Zhang, Shuangjia Zheng, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09496)  

**Abstract**: Designing protein sequences with optimal energetic stability is a key challenge in protein inverse folding, as current deep learning methods are primarily trained by maximizing sequence recovery rates, often neglecting the energy of the generated sequences. This work aims to overcome this limitation by developing a model that directly generates low-energy, stable protein sequences. We propose EnerBridge-DPO, a novel inverse folding framework focused on generating low-energy, high-stability protein sequences. Our core innovation lies in: First, integrating Markov Bridges with Direct Preference Optimization (DPO), where energy-based preferences are used to fine-tune the Markov Bridge model. The Markov Bridge initiates optimization from an information-rich prior sequence, providing DPO with a pool of structurally plausible sequence candidates. Second, an explicit energy constraint loss is introduced, which enhances the energy-driven nature of DPO based on prior sequences, enabling the model to effectively learn energy representations from a wealth of prior knowledge and directly predict sequence energy values, thereby capturing quantitative features of the energy landscape. Our evaluations demonstrate that EnerBridge-DPO can design protein complex sequences with lower energy while maintaining sequence recovery rates comparable to state-of-the-art models, and accurately predicts $\Delta \Delta G$ values between various sequences. 

**Abstract (ZH)**: 设计具有最优能量稳定性的蛋白质序列是蛋白质逆折叠中的一个关键挑战，当前的深度学习方法主要通过最大化序列恢复率来训练，往往忽视了生成序列的能量。本工作旨在通过开发一个能够直接生成低能稳定蛋白质序列的模型来克服这一限制。我们提出了一种名为EnerBridge-DPO的新型逆折叠框架，专注于生成低能高稳定性的蛋白质序列。我们的核心创新在于：首先，将马尔可夫桥与直接偏好优化（DPO）相结合，使用基于能量的偏好来精细调整马尔可夫桥模型。马尔可夫桥从一个信息丰富的先验序列开始优化，为DPO提供一组结构上可行的序列候选。其次，引入了一个明确的能量约束损失，这增强了基于先验序列的DPO的能量驱动性质，使模型能够从丰富的先验知识中有效学习能量表示，并直接预测序列的能量值，从而捕捉能量景观的定量特征。我们的评估表明，EnerBridge-DPO可以在保持与最新模型相当的序列恢复率的同时，设计出能量较低的蛋白质复合序列，并准确预测不同序列之间的$\Delta \Delta G$值。 

---
# BemaGANv2: A Tutorial and Comparative Survey of GAN-based Vocoders for Long-Term Audio Generation 

**Title (ZH)**: BemaGANv2：基于GAN的长时音频生成 vocoder 的教程与比较综述 

**Authors**: Taesoo Park, Mungwi Jeong, Mingyu Park, Narae Kim, Junyoung Kim, Mujung Kim, Jisang Yoo, Hoyun Lee, Sanghoon Kim, Soonchul Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.09487)  

**Abstract**: This paper presents a tutorial-style survey and implementation guide of BemaGANv2, an advanced GAN-based vocoder designed for high-fidelity and long-term audio generation. Built upon the original BemaGAN architecture, BemaGANv2 incorporates major architectural innovations by replacing traditional ResBlocks in the generator with the Anti-aliased Multi-Periodicity composition (AMP) module, which internally applies the Snake activation function to better model periodic structures. In the discriminator framework, we integrate the Multi-Envelope Discriminator (MED), a novel architecture we originally proposed, to extract rich temporal envelope features crucial for periodicity detection. Coupled with the Multi-Resolution Discriminator (MRD), this combination enables more accurate modeling of long-range dependencies in audio. We systematically evaluate various discriminator configurations, including MSD + MED, MSD + MRD, and MPD + MED + MRD, using objective metrics (FAD, SSIM, PLCC, MCD) and subjective evaluations (MOS, SMOS). This paper also provides a comprehensive tutorial on the model architecture, training methodology, and implementation to promote reproducibility. The code and pre-trained models are available at: this https URL. 

**Abstract (ZH)**: 本文提供了BemaGANv2的教程式综述与实现指南，BemaGANv2是基于GAN的 vocoder，旨在实现高保真度和长时间音频生成。BemaGANv2在原始BemaGAN架构的基础上引入了重大架构创新，通过使用抗混叠多周期组成（AMP）模块替代生成器中的传统ResBlock，并在内部应用蛇形激活函数以更好地建模周期结构。在判别框架中，我们结合了我们最初提出的一种新颖架构——多包络判别器（MED），以提取对于周期性检测至关重要的丰富时间包络特征。结合多分辨率判别器（MRD），这一组合能够更准确地建模音频中的长期依赖性。我们使用客观指标（FAD、SSIM、PLCC、MCD）和主观评估（MOS、SMOS）系统性地评估了各种判别器配置。本文还提供了模型架构、训练方法和实现的全面教程，以促进可再现性。可从以下链接获取代码和预训练模型：this https URL。 

---
# Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation 

**Title (ZH)**: Adv-BMT: 双向运动变换器用于安全关键交通场景生成 

**Authors**: Yuxin Liu, Zhenghao Peng, Xuanhao Cui, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09485)  

**Abstract**: Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstruct the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20\% compared to previous work. 

**Abstract (ZH)**: 基于场景的测试对于验证自动驾驶（AD）系统的性能至关重要。然而，现有的现实世界收集的数据集中缺乏长尾的安全关键场景，限制了此类测试。为解决数据问题，我们提出了Adv-BMT框架，该框架通过引入多样且现实的对抗性相互作用来扩展现实世界的场景。Adv-BMT的核心组件是一个双向运动变换器（BMT）模型，该模型用于执行逆向交通运动预测，它以场景中最后一个时间步的代理信息为输入，并以逆时间顺序重建交通，直到初始时间步。Adv-BMT框架是一个两阶段的管道：首先进行对抗性初始化，然后进行逆运动预测。与以往工作不同，我们不需要任何碰撞数据进行预训练，并且能够生成真实的多样化的碰撞交互。实验结果验证了Adv-BMT生成的碰撞场景的质量：在我们扩充的数据集中进行训练与以往工作相比，能使场景中的碰撞率降低20%。 

---
# Abstraction-Based Proof Production in Formal Verification of Neural Networks 

**Title (ZH)**: 基于抽象的证明生成在神经网络的形式验证中 

**Authors**: Yizhak Yisrael Elboher, Omri Isac, Guy Katz, Tobias Ladner, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09455)  

**Abstract**: Modern verification tools for deep neural networks (DNNs) increasingly rely on abstraction to scale to realistic architectures. In parallel, proof production is becoming a critical requirement for increasing the reliability of DNN verification results. However, current proofproducing verifiers do not support abstraction-based reasoning, creating a gap between scalability and provable guarantees. We address this gap by introducing a novel framework for proof-producing abstraction-based DNN verification. Our approach modularly separates the verification task into two components: (i) proving the correctness of an abstract network, and (ii) proving the soundness of the abstraction with respect to the original DNN. The former can be handled by existing proof-producing verifiers, whereas we propose the first method for generating formal proofs for the latter. This preliminary work aims to enable scalable and trustworthy verification by supporting common abstraction techniques within a formal proof framework. 

**Abstract (ZH)**: 现代深度神经网络(DNN)验证工具越来越多地依赖抽象来扩展到现实架构。与此同时，证明生产正逐渐成为提高DNN验证结果可靠性的关键要求。然而，当前的证明生产验证器不支持基于抽象的推理，从而在可扩展性和可证明保证之间造成了差距。我们通过引入一种新的基于抽象的DNN验证的证明生产框架来解决这一差距。我们的方法将验证任务模块化地分为两个部分：（i）证明抽象网络的正确性，以及（ii）证明抽象相对于原始DNN的正确性。前者可以由现有的证明生产验证器处理，而我们则提出了生成形式证明以处理后者的第一种方法。本初步工作旨在通过在形式证明框架中支持常见的抽象技术，实现可扩展和可信赖的验证。 

---
# UniToMBench: Integrating Perspective-Taking to Improve Theory of Mind in LLMs 

**Title (ZH)**: UniToMBench: 将换位思考整合以提升预训练语言模型的理论思维能力 

**Authors**: Prameshwar Thiyagarajan, Vaishnavi Parimi, Shamant Sai, Soumil Garg, Zhangir Meirbek, Nitin Yarlagadda, Kevin Zhu, Chris Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09450)  

**Abstract**: Theory of Mind (ToM), the ability to understand the mental states of oneself and others, remains a challenging area for large language models (LLMs), which often fail to predict human mental states accurately. In this paper, we introduce UniToMBench, a unified benchmark that integrates the strengths of SimToM and TOMBENCH to systematically improve and assess ToM capabilities in LLMs by integrating multi-interaction task designs and evolving story scenarios. Supported by a custom dataset of over 1,000 hand-written scenarios, UniToMBench combines perspective-taking techniques with diverse evaluation metrics to better stimulate social cognition in LLMs. Through evaluation, we observe that while models like GPT-4o and GPT-4o Mini show consistently high accuracy in tasks involving emotional and belief-related scenarios, with results usually above 80%, there is significant variability in their performance across knowledge-based tasks. These results highlight both the strengths and limitations of current LLMs in ToM-related tasks, underscoring the value of UniToMBench as a comprehensive tool for future development. Our code is publicly available here: this https URL. 

**Abstract (ZH)**: 理论心智（ToM）：统一基准UniToMBench通过集成SimToM和TOMBENCH的优势，系统地提升和评估LLMs的ToM能力，融合多交互任务设计和演进的故事场景。 

---
# TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision 

**Title (ZH)**: TOGA：基于时间的开放性视频QA弱监督方法 

**Authors**: Ayush Gupta, Anirban Roy, Rama Chellappa, Nathaniel D. Bastian, Alvaro Velasquez, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2506.09445)  

**Abstract**: We address the problem of video question answering (video QA) with temporal grounding in a weakly supervised setup, without any temporal annotations. Given a video and a question, we generate an open-ended answer grounded with the start and end time. For this task, we propose TOGA: a vision-language model for Temporally Grounded Open-Ended Video QA with Weak Supervision. We instruct-tune TOGA to jointly generate the answer and the temporal grounding. We operate in a weakly supervised setup where the temporal grounding annotations are not available. We generate pseudo labels for temporal grounding and ensure the validity of these labels by imposing a consistency constraint between the question of a grounding response and the response generated by a question referring to the same temporal segment. We notice that jointly generating the answers with the grounding improves performance on question answering as well as grounding. We evaluate TOGA on grounded QA and open-ended QA tasks. For grounded QA, we consider the NExT-GQA benchmark which is designed to evaluate weakly supervised grounded question answering. For open-ended QA, we consider the MSVD-QA and ActivityNet-QA benchmarks. We achieve state-of-the-art performance for both tasks on these benchmarks. 

**Abstract (ZH)**: 我们在弱监督下解决带有时间定位的视频问答问题，无需任何时间标注。给定一个视频和一个问题，我们生成一个开放式答案，并标注起始和结束时间。为此任务，我们提出TOGA：一种用于弱监督下带有时间定位的开放式视频问答的视觉-语言模型。我们在一个缺乏时间标注的时间定位弱监督设置中指令调用来同时生成答案和时间定位。我们通过在定位响应的问题与指向同一时间段的相同问题生成的回答之间施加一致性约束来为时间定位生成伪标签，确保这些标签的有效性。我们注意到同时生成答案与定位可以提高问答以及定位的性能。我们分别在带有定位的问答和开放式问答任务上评估TOGA。对于带有定位的问答，我们考虑了设计用于评估弱监督下带有定位的问答的NExT-GQA基准。对于开放式问答，我们考虑了MSVD-QA和ActivityNet-QA基准。我们在这些基准上分别实现了这两种任务的最佳性能。 

---
# GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture 

**Title (ZH)**: gigachat 家族：通过专家混合架构高效构建俄语语言模型 

**Authors**: GigaChat team, Mamedov Valentin, Evgenii Kosarev, Gregory Leleytner, Ilya Shchuckin, Valeriy Berezovskiy, Daniil Smirnov, Dmitry Kozlov, Sergei Averkiev, Lukyanenko Ivan, Aleksandr Proshunin, Ainur Israfilova, Ivan Baskov, Artem Chervyakov, Emil Shakirov, Mikhail Kolesov, Daria Khomich, Darya Latortseva, Sergei Porkhun, Yury Fedorov, Oleg Kutuzov, Polina Kudriavtseva, Sofiia Soldatova, Kolodin Egor, Stanislav Pyatkin, Dzmitry Menshykh, Grafov Sergei, Eldar Damirov, Karlov Vladimir, Ruslan Gaitukiev, Arkadiy Shatenov, Alena Fenogenova, Nikita Savushkin, Fedor Minkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09440)  

**Abstract**: Generative large language models (LLMs) have become crucial for modern NLP research and applications across various languages. However, the development of foundational models specifically tailored to the Russian language has been limited, primarily due to the significant computational resources required. This paper introduces the GigaChat family of Russian LLMs, available in various sizes, including base models and instruction-tuned versions. We provide a detailed report on the model architecture, pre-training process, and experiments to guide design choices. In addition, we evaluate their performance on Russian and English benchmarks and compare GigaChat with multilingual analogs. The paper presents a system demonstration of the top-performing models accessible via an API, a Telegram bot, and a Web interface. Furthermore, we have released three open GigaChat models in open-source (this https URL), aiming to expand NLP research opportunities and support the development of industrial solutions for the Russian language. 

**Abstract (ZH)**: 生成型大型语言模型（LLMs）已成为现代多语言NLP研究和应用的关键。然而，针对俄罗斯语言的基础模型开发受到限制，主要原因是对计算资源的需求巨大。本文介绍了GigaChat家族系列俄语LLMs，包括不同规模的基础模型和指令调优版本，并提供了详细的模型架构、预训练过程及实验结果，以指导设计选择。此外，我们在俄语和英语基准上评估了这些模型的性能，并将其与多语言对照组进行了比较。本文还展示了顶级模型通过API、Telegram bot和Web界面的系统演示。我们还发布了三个开源的GigaChat模型（请参见以下链接），以扩大NLP研究机会，并支持俄语语言的工业解决方案开发。 

---
# When Is Diversity Rewarded in Cooperative Multi-Agent Learning? 

**Title (ZH)**: 在合作多智能体学习中，何时会奖励多样性？ 

**Authors**: Michael Amir, Matteo Bettini, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2506.09434)  

**Abstract**: The success of teams in robotics, nature, and society often depends on the division of labor among diverse specialists; however, a principled explanation for when such diversity surpasses a homogeneous team is still missing. Focusing on multi-agent task allocation problems, our goal is to study this question from the perspective of reward design: what kinds of objectives are best suited for heterogeneous teams? We first consider an instantaneous, non-spatial setting where the global reward is built by two generalized aggregation operators: an inner operator that maps the $N$ agents' effort allocations on individual tasks to a task score, and an outer operator that merges the $M$ task scores into the global team reward. We prove that the curvature of these operators determines whether heterogeneity can increase reward, and that for broad reward families this collapses to a simple convexity test. Next, we ask what incentivizes heterogeneity to emerge when embodied, time-extended agents must learn an effort allocation policy. To study heterogeneity in such settings, we use multi-agent reinforcement learning (MARL) as our computational paradigm, and introduce Heterogeneous Environment Design (HED), a gradient-based algorithm that optimizes the parameter space of underspecified MARL environments to find scenarios where heterogeneity is advantageous. Experiments in matrix games and an embodied Multi-Goal-Capture environment show that, despite the difference in settings, HED rediscovers the reward regimes predicted by our theory to maximize the advantage of heterogeneity, both validating HED and connecting our theoretical insights to reward design in MARL. Together, these results help us understand when behavioral diversity delivers a measurable benefit. 

**Abstract (ZH)**: 机器人学、自然和社会中团队的成功 often取决于多样专家之间的劳动分工；然而，关于这种多样性何时超越了 homogeneous 团队的规范解释仍然缺失。我们关注多智能体任务分配问题，从奖励设计的角度研究这一问题：什么样的目标最适合 heterogeneous 团队？我们首先考虑一个瞬时、非空间性的设置，其中全局奖励由两个广义聚合操作员构建：一个内操作员将 N 个智能体在单个任务上的努力分配映射到任务分数，另一个外操作员将 M 个任务分数合并成全局团队奖励。我们证明这些操作员的曲率决定了多样性是否能增加奖励，并且对于广泛的奖励家庭，这归结为一个简单的凸性测试。接下来，我们探讨当具身化的、时间延伸的智能体必须学习努力分配策略时，是什么激励了多样性的出现。为了研究这种设置下的多样性，我们以多智能体强化学习（MARL）作为计算范式，并引入多样环境设计（HED），这是一种基于梯度的算法，在欠规定的 MARL 环境的参数空间中进行优化，以找到多样化有益的场景。在矩阵游戏和一个具身的多目标捕捉环境中进行的实验表明，尽管设置不同，HED 重新发现了我们的理论预测的最大化多样性的奖励区域，这验证了 HED 并将我们的理论洞察与 MARL 中的奖励设计联系起来。总之，这些结果帮助我们理解当行为多样性带来可度量的好处时。 

---
# Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting 

**Title (ZH)**: 改进的监督 fine-tuning 方法以减轻大型语言模型的灾难性遗忘 

**Authors**: Fei Ding, Baiqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09428)  

**Abstract**: Supervised Fine-Tuning (SFT), while enhancing large language models(LLMs)' instruction-following capabilities and domain-specific task adaptability, often diminishes their general capabilities. Moreover, due to the inaccessibility of original pre-training data, catastrophic forgetting tends to be exacerbated when third-party practitioners implement SFT on open-sourced models. To address this challenge, we propose a novel, more cost-effective SFT method which could effectively reduce the risk of catastrophic forgetting without access to original SFT data. Our approach begins by reconstructing the likely SFT instruction distribution of the base model, followed by a multi-model screening process to select optimal data, which is then mixed with new data for SFT. Experimental results demonstrate that our method preserves generalization capabilities in general domains while improving task-specific performance. 

**Abstract (ZH)**: 监督微调（SFT）虽然增强了大型语言模型（LLMs）的指令遵循能力和领域特定任务的适应性，但往往会损害其通用能力。此外，由于无法访问原始预训练数据，第三方实践者在开源模型上实施SFT时，灾难性遗忘的问题往往会加剧。为此，我们提出了一种新颖且成本更低的SFT方法，能够在不访问原始SFT数据的情况下有效降低灾难性遗忘的风险。该方法首先重构基模型可能的SFT指令分布，然后通过多模型筛选过程选择最优数据，将其与新数据混合以进行SFT。实验结果表明，该方法在保持通用域上的泛化能力的同时，提高了任务特定性能。 

---
# A High-Quality Dataset and Reliable Evaluation for Interleaved Image-Text Generation 

**Title (ZH)**: 高质量数据集与可靠的 interleaved 图像-文本 生成评估 

**Authors**: Yukang Feng, Jianwen Sun, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Yifan Chang, Sizhuo Zhou, Shenglin Zhang, Yu Dai, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09427)  

**Abstract**: Recent advancements in Large Multimodal Models (LMMs) have significantly improved multimodal understanding and generation. However, these models still struggle to generate tightly interleaved image-text outputs, primarily due to the limited scale, quality and instructional richness of current training datasets. To address this, we introduce InterSyn, a large-scale multimodal dataset constructed using our Self-Evaluation with Iterative Refinement (SEIR) method. InterSyn features multi-turn, instruction-driven dialogues with tightly interleaved imagetext responses, providing rich object diversity and rigorous automated quality refinement, making it well-suited for training next-generation instruction-following LMMs. Furthermore, to address the lack of reliable evaluation tools capable of assessing interleaved multimodal outputs, we introduce SynJudge, an automatic evaluation model designed to quantitatively assess multimodal outputs along four dimensions: text content, image content, image quality, and image-text synergy.
Experimental studies show that the SEIR method leads to substantially higher dataset quality compared to an otherwise identical process without refinement.
Moreover, LMMs trained on InterSyn achieve uniform performance gains across all evaluation metrics, confirming InterSyn's utility for advancing multimodal systems. 

**Abstract (ZH)**: 近期大型多模态模型（LMMs）的进展显著提高了多模态的理解和生成能力，但这些模型仍然难以生成紧密交织的图像-文本输出，主要原因是当前训练数据集的规模、质量和指导丰富性有限。为解决这一问题，我们引入了InterSyn，这是一种使用我们自评价与迭代细化（SEIR）方法构建的大规模多模态数据集。InterSyn 包含多轮、指令驱动的对话，具有紧密交织的图像-文本响应，提供丰富的对象多样性并进行严格的自动化质量 refinement，使其非常适合训练下一代遵循指令的LMMs。此外，为了解决缺乏可靠的评估工具来评估交织的多模态输出的问题，我们引入了SynJudge，这是一种自动评估模型，旨在从文本内容、图像内容、图像质量和图像-文本协同作用四个维度定量评估多模态输出。实验研究显示，SEIR方法导致了显著更高的数据集质量。此外，使用InterSyn训练的LMMs在整个评估指标上均实现了均匀的性能提升，进一步证明了InterSyn对推动多模态系统发展的实用性。 

---
# Synthetic Human Action Video Data Generation with Pose Transfer 

**Title (ZH)**: 合成人体动作视频数据生成与姿态转移 

**Authors**: Vaclav Knapp, Matyas Bohacek  

**Link**: [PDF](https://arxiv.org/pdf/2506.09411)  

**Abstract**: In video understanding tasks, particularly those involving human motion, synthetic data generation often suffers from uncanny features, diminishing its effectiveness for training. Tasks such as sign language translation, gesture recognition, and human motion understanding in autonomous driving have thus been unable to exploit the full potential of synthetic data. This paper proposes a method for generating synthetic human action video data using pose transfer (specifically, controllable 3D Gaussian avatar models). We evaluate this method on the Toyota Smarthome and NTU RGB+D datasets and show that it improves performance in action recognition tasks. Moreover, we demonstrate that the method can effectively scale few-shot datasets, making up for groups underrepresented in the real training data and adding diverse backgrounds. We open-source the method along with RANDOM People, a dataset with videos and avatars of novel human identities for pose transfer crowd-sourced from the internet. 

**Abstract (ZH)**: 视频理解任务中，特别是在人体动作涉及的任务中，合成数据生成往往存在不自然的特点，这降低了其训练效果。诸如手语翻译、姿态识别和自动驾驶中的人体动作理解等任务因此未能充分利用合成数据的潜力。本文提出了一种使用姿态转移（具体来说是可控的3D高斯 avatar模型）生成合成人体动作视频数据的方法。我们在Toyota Smarthome和NTU RGB+D数据集上评估了该方法，并展示了其在动作识别任务中的性能提升。此外，我们证明该方法可以有效扩大少样本数据集的规模，弥补现实训练数据中代表性不足的群体，并增加多样化的背景环境。我们开源了该方法以及从互联网中 crowdsourced 来的包含新颖人体身份视频和avatar的RANDOM People数据集。 

---
# Token Constraint Decoding Improves Robustness on Question Answering for Large Language Models 

**Title (ZH)**: Token Constraint Decoding 提高大型语言模型在问答任务上的鲁棒性 

**Authors**: Jui-Ming Yao, Hao-Yuan Chen, Zi-Xian Tang, Bing-Jia Tan, Sheng-Wei Peng, Bing-Cheng Xie, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.09408)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance on multiple-choice question answering (MCQA) benchmarks, yet they remain highly vulnerable to minor input perturbations. In this paper, we introduce and evaluate Token Constraint Decoding (TCD). This simple yet effective inference-time algorithm enforces alignment between token-level predictions to enhance robustness in noisy settings. Through extensive experiments on CommonsenseQA, MMLU, and MMLU-Pro, we show that TCD, especially when paired with prompt engineering (PE) fixes, significantly restores performance degraded by input noise, yielding up to +39\% absolute gains for weaker models like Gemma3 1B. Penalty sweep analyses further reveal that TCD implicitly regularizes overconfident outputs, with different models requiring distinct penalty schedules to maximize resilience. Our findings establish TCD as a practical, model-agnostic approach for improving reasoning stability under real-world imperfections and pave the way for more reliable deployment of LLMs in safety-critical or user-facing applications. 

**Abstract (ZH)**: Large Language Models (LLMs) 在多项选择题回答（MCQA）基准测试中展现了令人印象深刻的性能，但却对细微的输入扰动极为敏感。本文我们介绍了并评估了 Token Constraint Decoding (TCD)。这一简单而有效的推理时算法在嘈杂环境中通过强制预测间的对齐来增强鲁棒性。我们在 CommonsenseQA、MMLU 和 MMLU-Pro 上进行了广泛的实验，表明 TCD，尤其是与提示工程 (PE) 修正结合使用时，可以显著恢复由输入噪声导致的性能下降，对于像 Gemma3 1B 这样较弱的模型，绝对收益可达 +39%。进一步的惩罚范围分析表明，TCD 会隐式地正则化过于自信的输出，不同模型需要不同的惩罚时间表以最大化鲁棒性。我们的研究结果确立了 TCD 作为一种实用且模型无关的方法，用于在现实世界的缺陷下提高推理稳定性，并为在关键安全或用户面向应用中更可靠部署 LLMs 打下了基础。 

---
# SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving 

**Title (ZH)**: SLED: 一种用于高效边缘服务的 speculative LLM 解码框架 

**Authors**: Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Dimitrios Nikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09397)  

**Abstract**: Regardless the advancements in device capabilities, efficient inferencing advanced large language models (LLMs) at the edge remains challenging due to limited device memory and power constraints. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new approach that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose SLED, a method that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server efficiently batches and verifies the tokens utilizing a more precise target model. This approach supports device heterogeneity and reduces server-side memory footprint by avoiding the need to deploy multiple target models. Our initial experiments with Jetson Orin Nano, Raspberry Pi 5, and an RTX 6000 edge server indicate substantial benefits: significantly reduced latency, improved energy efficiency, and increased concurrent inference sessions, all without sacrificing model accuracy. 

**Abstract (ZH)**: 无论设备能力如何提升，由于有限的设备内存和功率约束，在边缘高效推理先进大型语言模型（LLMs）仍然具有挑战性。现有策略，如激进量化、剪枝或远程推理，要么牺牲准确性，要么导致成本负担增加。本文介绍了一种新方法，该方法利用了以前主要被视为自回归生成加速技术的推测解码，作为专门为边缘计算设计的方法，通过跨异构设备协调计算。我们提出了一种名为SLED的方法，该方法允许轻量级边缘设备使用多种草稿模型在当地生成多个候选词汇，而单个共享边缘服务器则有效批处理和验证词汇，利用更精确的目标模型进行验证。该方法支持设备异构性，并通过避免部署多个目标模型来减少服务器端的内存占用。初步实验结果表明，这种方法在Jetson Orin Nano、Raspberry Pi 5和RTX 6000边缘服务器上带来了显著的益处：显著降低延迟、提高能效并增加并发推理会话，同时无需牺牲模型准确性。 

---
# Reasoning as a Resource: Optimizing Fast and Slow Thinking in Code Generation Models 

**Title (ZH)**: 将推理作为一种资源：优化代码生成模型中的快速与慢速思考 

**Authors**: Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09396)  

**Abstract**: This position paper proposes a fundamental shift in designing code generation models: treating reasoning depth as a controllable resource. Rather than being an incidental byproduct of prompting, we argue that the trade-off between rapid, direct answers ("fast thinking") and elaborate, chain-of-thought deliberation ("slow thinking") must be explicitly managed. We contend that optimizing reasoning budgets across the entire model lifecycle - from synthetic data creation and benchmarking to real-world deploymen - can unlock superior trade-offs among accuracy, latency, and cost. This paper outlines how adaptive control over reasoning can enrich supervision signals, motivate new multi-dimensional benchmarks, and inform cost-aware, security-conscious deployment policies. By viewing fast and slow thinking as complementary modes to be scheduled, we envision coding agents that think deep when necessary and act fast when possible. 

**Abstract (ZH)**: 本论点论文提出在设计代码生成模型时进行根本性的转变：将推理深度视为可控资源。我们认为，快速直接的答案（“快速思考”）与详细逐步的权衡决策（“慢速思考”）之间的权衡不应被视为提示的附带产物，而应明确管理。我们主张在整个模型生命周期中优化推理预算——从合成数据创建和基准测试到实际部署——以解锁在准确率、延迟和成本之间的更优权衡。本文介绍了如何通过适应性控制推理来丰富监督信号、激发新的多维基准测试，并指导成本意识与安全意识的部署策略。通过将快速与慢速思考视为可调度的互补方式，我们设想能够根据需要深入思考并在可能的情况下快速行动的编码代理。 

---
# Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations 

**Title (ZH)**: 双足平衡控制的全身肌骨站立与跌倒仿真研究 

**Authors**: Chengtian Ma, Yunyue Wei, Chenhui Zuo, Chen Zhang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09383)  

**Abstract**: Balance control is important for human and bipedal robotic systems. While dynamic balance during locomotion has received considerable attention, quantitative understanding of static balance and falling remains limited. This work presents a hierarchical control pipeline for simulating human balance via a comprehensive whole-body musculoskeletal system. We identified spatiotemporal dynamics of balancing during stable standing, revealed the impact of muscle injury on balancing behavior, and generated fall contact patterns that aligned with clinical data. Furthermore, our simulated hip exoskeleton assistance demonstrated improvement in balance maintenance and reduced muscle effort under perturbation. This work offers unique muscle-level insights into human balance dynamics that are challenging to capture experimentally. It could provide a foundation for developing targeted interventions for individuals with balance impairments and support the advancement of humanoid robotic systems. 

**Abstract (ZH)**: 平衡控制对于人类和双足机器人系统至关重要。虽然在运动过程中的动态平衡受到了广泛关注，但静态平衡和摔倒的量化理解仍然有限。本研究提出了一种分层控制管道，通过综合全身肌肉骨骼系统模拟人类平衡。我们识别了稳定站立期间平衡的时空动态，揭示了肌肉损伤对平衡行为的影响，并生成了与临床数据相一致的摔倒接触模式。此外，我们模拟的髋部外骨骼辅助表明，在外部扰动下平衡维持的改善和肌肉努力的减少。本研究提供了难以通过实验捕捉到的人类平衡动力学的肌肉层面见解，可为发展针对平衡障碍个体的靶向干预措施以及促进类人机器人系统的发展奠定基础。 

---
# LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization 

**Title (ZH)**: LPO：通过位置偏好优化实现精确的GUI代理交互 

**Authors**: Jiaqi Tang, Yu Xia, Yi-Feng Wu, Yuwei Hu, Yuhui Chen, Qing-Guo Chen, Xiaogang Xu, Xiangyu Wu, Hao Lu, Yanqing Ma, Shiyin Lu, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09373)  

**Abstract**: The advent of autonomous agents is transforming interactions with Graphical User Interfaces (GUIs) by employing natural language as a powerful intermediary. Despite the predominance of Supervised Fine-Tuning (SFT) methods in current GUI agents for achieving spatial localization, these methods face substantial challenges due to their limited capacity to accurately perceive positional data. Existing strategies, such as reinforcement learning, often fail to assess positional accuracy effectively, thereby restricting their utility. In response, we introduce Location Preference Optimization (LPO), a novel approach that leverages locational data to optimize interaction preferences. LPO uses information entropy to predict interaction positions by focusing on zones rich in information. Besides, it further introduces a dynamic location reward function based on physical distance, reflecting the varying importance of interaction positions. Supported by Group Relative Preference Optimization (GRPO), LPO facilitates an extensive exploration of GUI environments and significantly enhances interaction precision. Comprehensive experiments demonstrate LPO's superior performance, achieving SOTA results across both offline benchmarks and real-world online evaluations. Our code will be made publicly available soon, at this https URL. 

**Abstract (ZH)**: 自主代理的出现正在通过采用自然语言作为强中介来转变与图形用户界面（GUI）的交互。尽管当前的GUI代理中监督微调（SFT）方法在实现空间定位方面占据主导地位，但由于其感知位置数据能力有限，这些方法面临着重大挑战。现有策略，如强化学习，往往难以有效地评估位置准确性，从而限制了其应用。为此，我们提出了位置偏好优化（LPO）这一新颖方法，该方法利用位置数据来优化交互偏好。LPO 使用信息熵来预测交互位置，重点关注信息丰富的区域。此外，LPO 还引入了基于物理距离的动态位置奖励函数，反映了交互位置的不同重要性。借助群组相对偏好优化（GRPO），LPO 促进了GUI环境的广泛探索，并显著提高了交互精度。全面的实验表明，LPO 在离线基准和实际在线评估中均实现了最先进的性能。我们的代码将很快公开，地址见此 https URL。 

---
# Anomaly Detection and Generation with Diffusion Models: A Survey 

**Title (ZH)**: 基于扩散模型的异常检测与生成：一个综述 

**Authors**: Yang Liu, Jing Liu, Chengfang Li, Rui Xi, Wenchao Li, Liang Cao, Jin Wang, Laurence T. Yang, Junsong Yuan, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09368)  

**Abstract**: Anomaly detection (AD) plays a pivotal role across diverse domains, including cybersecurity, finance, healthcare, and industrial manufacturing, by identifying unexpected patterns that deviate from established norms in real-world data. Recent advancements in deep learning, specifically diffusion models (DMs), have sparked significant interest due to their ability to learn complex data distributions and generate high-fidelity samples, offering a robust framework for unsupervised AD. In this survey, we comprehensively review anomaly detection and generation with diffusion models (ADGDM), presenting a tutorial-style analysis of the theoretical foundations and practical implementations and spanning images, videos, time series, tabular, and multimodal data. Crucially, unlike existing surveys that often treat anomaly detection and generation as separate problems, we highlight their inherent synergistic relationship. We reveal how DMs enable a reinforcing cycle where generation techniques directly address the fundamental challenge of anomaly data scarcity, while detection methods provide critical feedback to improve generation fidelity and relevance, advancing both capabilities beyond their individual potential. A detailed taxonomy categorizes ADGDM methods based on anomaly scoring mechanisms, conditioning strategies, and architectural designs, analyzing their strengths and limitations. We final discuss key challenges including scalability and computational efficiency, and outline promising future directions such as efficient architectures, conditioning strategies, and integration with foundation models (e.g., visual-language models and large language models). By synthesizing recent advances and outlining open research questions, this survey aims to guide researchers and practitioners in leveraging DMs for innovative AD solutions across diverse applications. 

**Abstract (ZH)**: 基于扩散模型的异常检测与生成综述（Anomaly Detection and Generation with Diffusion Models: A Survey） 

---
# COGENT: A Curriculum-oriented Framework for Generating Grade-appropriate Educational Content 

**Title (ZH)**: COGENT: 以课程为导向的生成适龄教育内容框架 

**Authors**: Zhengyuan Liu, Stella Xin Yin, Dion Hoe-Lian Goh, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09367)  

**Abstract**: While Generative AI has demonstrated strong potential and versatility in content generation, its application to educational contexts presents several challenges. Models often fail to align with curriculum standards and maintain grade-appropriate reading levels consistently. Furthermore, STEM education poses additional challenges in balancing scientific explanations with everyday language when introducing complex and abstract ideas and phenomena to younger students. In this work, we propose COGENT, a curriculum-oriented framework for generating grade-appropriate educational content. We incorporate three curriculum components (science concepts, core ideas, and learning objectives), control readability through length, vocabulary, and sentence complexity, and adopt a ``wonder-based'' approach to increase student engagement and interest. We conduct a multi-dimensional evaluation via both LLM-as-a-judge and human expert analysis. Experimental results show that COGENT consistently produces grade-appropriate passages that are comparable or superior to human references. Our work establishes a viable approach for scaling adaptive and high-quality learning resources. 

**Abstract (ZH)**: 基于课程的生成式教育内容框架：COGENT 

---
# SAGE: Exploring the Boundaries of Unsafe Concept Domain with Semantic-Augment Erasing 

**Title (ZH)**: SAGE: 探索语义增强擦除在不安全概念领域边界上的应用 

**Authors**: Hongguang Zhu, Yunchao Wei, Mengyu Wang, Siyu Jiao, Yan Fang, Jiannan Huang, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09363)  

**Abstract**: Diffusion models (DMs) have achieved significant progress in text-to-image generation. However, the inevitable inclusion of sensitive information during pre-training poses safety risks, such as unsafe content generation and copyright infringement. Concept erasing finetunes weights to unlearn undesirable concepts, and has emerged as a promising solution. However, existing methods treat unsafe concept as a fixed word and repeatedly erase it, trapping DMs in ``word concept abyss'', which prevents generalized concept-related erasing. To escape this abyss, we introduce semantic-augment erasing which transforms concept word erasure into concept domain erasure by the cyclic self-check and self-erasure. It efficiently explores and unlearns the boundary representation of concept domain through semantic spatial relationships between original and training DMs, without requiring additional preprocessed data. Meanwhile, to mitigate the retention degradation of irrelevant concepts while erasing unsafe concepts, we further propose the global-local collaborative retention mechanism that combines global semantic relationship alignment with local predicted noise preservation, effectively expanding the retentive receptive field for irrelevant concepts. We name our method SAGE, and extensive experiments demonstrate the comprehensive superiority of SAGE compared with other methods in the safe generation of DMs. The code and weights will be open-sourced at this https URL. 

**Abstract (ZH)**: 差分模型（DMs）在文本到图像生成方面取得了显著进展。然而，预训练过程中不可避免地包含敏感信息导致了安全风险，如不安全内容生成和版权侵犯。概念擦除微调权重以消除不良概念，并已成为一种有前景的解决方案。然而，现有方法将不安全概念视为固定词，并反复擦除它，将DMs困在“词汇概念深渊”中，这阻碍了对泛化概念相关擦除的能力。为了逃离这个深渊，我们引入了语义增强擦除，通过循环自检和自擦除将概念词擦除转化为概念领域擦除。它通过原始和训练DMs之间的语义空间关系高效地探索和遗忘概念领域的边界表示，无需额外预处理数据。同时，为在擦除不安全概念时减轻与之无关的概念保留降级，我们进一步提出了一种全局-局部协作保留机制，该机制结合了全局语义关系对齐与局部预测噪声保留，有效地扩展了与无关概念相关的保留感受野。我们称我们的方法为SAGE，并且广泛的实验表明，在DMs的安全生成方面，SAGE相比其他方法具有全面的优势。代码和权重将在该网址开放源代码。 

---
# "I Said Things I Needed to Hear Myself": Peer Support as an Emotional, Organisational, and Sociotechnical Practice in Singapore 

**Title (ZH)**: “我说了自己需要听到的话”：朋辈支持在新加坡作为情感的、组织的和社会技术的实践 

**Authors**: Kellie Yu Hui Sim, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09362)  

**Abstract**: Peer support plays a vital role in expanding access to mental health care by providing empathetic, community-based support outside formal clinical systems. As digital platforms increasingly mediate such support, the design and impact of these technologies remain under-examined, particularly in Asian contexts. This paper presents findings from an interview study with 20 peer supporters in Singapore, who operate across diverse online, offline, and hybrid environments. Through a thematic analysis, we unpack how participants start, conduct, and sustain peer support, highlighting their motivations, emotional labour, and the sociocultural dimensions shaping their practices. Building on this grounded understanding, we surface design directions for culturally responsive digital tools that scaffold rather than supplant relational care. Drawing insights from qualitative accounts, we offer a situated perspective on how AI might responsibly augment peer support. This research contributes to human-centred computing by articulating the lived realities of peer supporters and proposing design implications for trustworthy and context-sensitive AI in mental health. 

**Abstract (ZH)**: 同伴支持在外形式化临床系统之外提供了具有同理心的社区支持，对于扩大心理健康服务的获取发挥着至关重要的作用。随着数字平台在提供此类支持方面发挥越来越大的作用，这些技术的设计及其影响仍需进一步考察，尤其是在亚洲背景下。本文基于对新加坡20名同伴支持者的研究访谈，探讨他们在多种线上、线下及混合环境中的支持方式。通过主题分析，我们揭示了参与者如何开展和维持同伴支持，强调了他们的动机、情感劳动及其社会文化背景对其实践的影响。基于这一扎根理解，我们提出了文化响应性的数字工具设计方向，旨在搭桥而非取代关系性关怀。结合定性叙述，我们提出了一种负责任地增强同伴支持的在地性视角。本研究通过阐述同伴支持者的生活现实并提出可信赖且情境敏感的AI设计建议，贡献于以人为本的计算科学。 

---
# "Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions 

**Title (ZH)**: “这是真正的人类同伴支持者吗？”：在LLM支持的互动中同伴支持者与专家之间的 mis 对齐 

**Authors**: Kellie Yu Hui Sim, Roy Ka-Wei Lee, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09354)  

**Abstract**: Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client, context-sensitive LLM-generated suggestions, and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 5 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care. 

**Abstract (ZH)**: AI驱动的解决方案扩展心理社会支持的 acces: 基于实证经验的同伴支持的机遇与挑战 

---
# Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation 

**Title (ZH)**: 自回归对抗后训练实时交互式视频生成 

**Authors**: Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09350)  

**Abstract**: Existing large-scale video generation models are computationally intensive, preventing adoption in real-time and interactive applications. In this work, we propose autoregressive adversarial post-training (AAPT) to transform a pre-trained latent video diffusion model into a real-time, interactive video generator. Our model autoregressively generates a latent frame at a time using a single neural function evaluation (1NFE). The model can stream the result to the user in real time and receive interactive responses as controls to generate the next latent frame. Unlike existing approaches, our method explores adversarial training as an effective paradigm for autoregressive generation. This not only allows us to design an architecture that is more efficient for one-step generation while fully utilizing the KV cache, but also enables training the model in a student-forcing manner that proves to be effective in reducing error accumulation during long video generation. Our experiments demonstrate that our 8B model achieves real-time, 24fps, streaming video generation at 736x416 resolution on a single H100, or 1280x720 on 8xH100 up to a minute long (1440 frames). Visit our research website at this https URL 

**Abstract (ZH)**: 现有的大型视频生成模型计算密集，阻碍了其在实时和交互应用中的采用。在本文中，我们提出了一种自回归对抗后训练（AAPT）方法，将预训练的潜空间视频扩散模型转换为实时交互式视频生成器。我们的模型使用单次神经函数评估（1NFE）逐帧自回归生成潜在帧，并能够实时流式传输结果并接收交互式响应作为控制生成下一帧的指令。与现有方法不同，我们的方法探索了对抗训练作为一种有效的自回归生成范式。这不仅使我们能够设计一种更高效的单步生成架构，充分利用KV缓存，还使我们能够以一种证明对减少长时间视频生成过程中错误累积有效的学生强迫训练模型的方式进行训练。我们的实验表明，我们的8B模型可以在单块H100上以736x416分辨率实现每秒24帧的实时流式传输视频生成，或者在8块H100上生成长达一分钟（1440帧）的1280x720分辨率视频。请访问我们的研究网站：[访问链接]。 

---
# ErrorEraser: Unlearning Data Bias for Improved Continual Learning 

**Title (ZH)**: ErrorEraser: 消除数据偏见以改善连续学习 

**Authors**: Xuemei Cao, Hanlin Gu, Xin Yang, Bingjun Wei, Haoyang Liang, Xiangkun Wang, Tianrui Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09347)  

**Abstract**: Continual Learning (CL) primarily aims to retain knowledge to prevent catastrophic forgetting and transfer knowledge to facilitate learning new tasks. Unlike traditional methods, we propose a novel perspective: CL not only needs to prevent forgetting, but also requires intentional this http URL arises from existing CL methods ignoring biases in real-world data, leading the model to learn spurious correlations that transfer and amplify across tasks. From feature extraction and prediction results, we find that data biases simultaneously reduce CL's ability to retain and transfer knowledge. To address this, we propose ErrorEraser, a universal plugin that removes erroneous memories caused by biases in CL, enhancing performance in both new and old tasks. ErrorEraser consists of two modules: Error Identification and Error Erasure. The former learns the probability density distribution of task data in the feature space without prior knowledge, enabling accurate identification of potentially biased samples. The latter ensures only erroneous knowledge is erased by shifting the decision space of representative outlier samples. Additionally, an incremental feature distribution learning strategy is designed to reduce the resource overhead during error identification in downstream tasks. Extensive experimental results show that ErrorEraser significantly mitigates the negative impact of data biases, achieving higher accuracy and lower forgetting rates across three types of CL methods. The code is available at this https URL. 

**Abstract (ZH)**: 持续学习（CL）主要旨在保留知识以防止灾难性遗忘，并转移知识以促进新任务的学习。与传统方法不同，我们提出了一种新的视角：CL不仅需要防止遗忘，还需要有意地消除由于偏见引起的错误记忆。这种偏见来自于现有CL方法忽略现实世界数据中的偏差，导致模型学习转移并放大错误的相关性。通过特征提取和预测结果，我们发现数据偏见同时降低了CL保留和转移知识的能力。为此，我们提出ErrorEraser，这是一种通用插件，它可以消除由CL中偏见引起的各种错误记忆，从而在新任务和旧任务中均提高性能。ErrorEraser由两个模块组成：错误识别和错误消除。前者在无需先验知识的情况下学习特征空间中任务数据的概率密度分布，从而能够准确识别潜在的偏倚样本。后者通过调整代表性离群样本的决策空间，确保仅消除错误的知识。此外，我们设计了增量特征分布学习策略，以减少错误识别在下游任务中的资源开销。广泛的实验结果显示，ErrorEraser显著减轻了数据偏见的负面影响，在三种类型CL方法中均实现了更高的准确率和更低的遗忘率。代码可在以下链接获取：this https URL。 

---
# Latent Multi-Head Attention for Small Language Models 

**Title (ZH)**: 潜在多头自注意力机制在小型语言模型中的应用 

**Authors**: Sushant Mehta, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2506.09342)  

**Abstract**: We present the first comprehensive study of latent multi-head attention (MLA) for small language models, revealing interesting efficiency-quality trade-offs. Training 30M-parameter GPT models on 100,000 synthetic stories, we benchmark three architectural variants: standard multi-head attention (MHA), MLA, and MLA with rotary positional embeddings (MLA+RoPE). Our key finding is that MLA+RoPE with half-rank latent dimensions (r = d/2) achieves a 45% KV-cache memory reduction while incurring only a 0.3% increase in validation loss (essentially matching MHA quality)- a Pareto improvement for memory constrained deployment. We further show that RoPE is crucial for MLA in small models: without it, MLA underperforms vanilla attention by 3-5%, but with RoPE, it surpasses vanilla by 2%. Inference benchmarks on NVIDIA A100 GPUs reveal that MLA with r=d/2 achieves a 1.4 times speedup over full-rank MLA while maintaining the memory savings. GPT-4 evaluations corroborate perplexity results, with ours achieving the highest quality scores (7.4/10) across grammar, creativity, and consistency metrics. Code and models will be released upon acceptance. 

**Abstract (ZH)**: 我们首次全面研究了小语言模型中的潜在多头注意力（MLA），揭示了有趣的效率-质量权衡。在100,000个合成故事上训练30M参数的GPT模型，我们基准测试了三种架构变体：标准多头注意力（MHA）、MLA以及带有旋转位置嵌入的MLA（MLA+RoPE）。我们的主要发现是，带有半秩潜在维度（r=d/2）的MLA+RoPE实现了45%的KV缓存内存减少，同时只增加了0.3%的验证损失（基本上与MHA质量相当）——这是内存受限部署的理想改进。我们还表明，对于小模型来说，RoPE对于MLA至关重要：没有RoPE，MLA比vanilla注意力低3-5%的性能，但有了RoPE，它比vanilla高出2%的性能。针对NVIDIA A100 GPU的推理基准测试显示，半秩MLA与全秩MLA相比，速度提高了1.4倍，同时保留了内存节省效果。GPT-4评估结果与困惑度结果相符，我们的模型在这几项指标（语法、创造力和一致性）中获得了最高质量评分（7.4/10）。接受后我们将发布代码和模型。 

---
# RePO: Replay-Enhanced Policy Optimization 

**Title (ZH)**: RePO: 回放增强策略优化 

**Authors**: Siheng Li, Zhanhui Zhou, Wai Lam, Chao Yang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09340)  

**Abstract**: Reinforcement learning (RL) is vital for optimizing large language models (LLMs). Recent Group Relative Policy Optimization (GRPO) estimates advantages using multiple on-policy outputs per prompt, leading to high computational costs and low data efficiency. To address this, we introduce Replay-Enhanced Policy Optimization (RePO), which leverages diverse replay strategies to retrieve off-policy samples from a replay buffer, allowing policy optimization based on a broader and more diverse set of samples for each prompt. Experiments on five LLMs across seven mathematical reasoning benchmarks demonstrate that RePO achieves absolute average performance gains of $18.4$ and $4.1$ points for Qwen2.5-Math-1.5B and Qwen3-1.7B, respectively, compared to GRPO. Further analysis indicates that RePO increases computational cost by $15\%$ while raising the number of effective optimization steps by $48\%$ for Qwen3-1.7B, with both on-policy and off-policy sample numbers set to $8$. The repository can be accessed at this https URL. 

**Abstract (ZH)**: 强化学习（RL）对于优化大型语言模型（LLMs）至关重要。近来的群组相对策略优化（GRPO）通过每个提示使用多个在线策略输出来估计优势，导致高计算成本和低数据效率。为了解决这一问题，我们引入了回放增强策略优化（RePO），该方法利用多样的回放策略从回放缓存中检索离策样例，使得每个提示基于更广泛和多样化的样本集进行策略优化。实验在五个LLM上针对七个数学推理基准表明，RePO分别在Qwen2.5-Math-1.5B和Qwen3-1.7B上与GRPO相比，实现了绝对平均性能提升18.4和4.1分点。进一步分析表明，对于Qwen3-1.7B，RePO将计算成本增加了15%，同时有效优化步数增加了48%，并且每种样本数量设置为8。代码库可以访问这里。 

---
# Know What You Don't Know: Uncertainty Calibration of Process Reward Models 

**Title (ZH)**: 知其所不知：过程奖励模型的不确定性校准 

**Authors**: Young-Jin Park, Kristjan Greenewald, Kaveh Alim, Hao Wang, Navid Azizan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09338)  

**Abstract**: Process reward models (PRMs) play a central role in guiding inference-time scaling algorithms for large language models (LLMs). However, we observe that even state-of-the-art PRMs can be poorly calibrated and often overestimate success probabilities. To address this, we present a calibration approach, performed via quantile regression, that adjusts PRM outputs to better align with true success probabilities. Leveraging these calibrated success estimates and their associated confidence bounds, we introduce an \emph{instance-adaptive scaling} (IAS) framework that dynamically adjusts the inference budget based on the estimated likelihood that a partial reasoning trajectory will yield a correct final answer. Unlike conventional methods that allocate a fixed number of reasoning trajectories per query, this approach successfully adapts to each instance and reasoning step when using our calibrated PRMs. Experiments on mathematical reasoning benchmarks show that (i) our PRM calibration method successfully achieves small calibration error, outperforming the baseline methods, (ii) calibration is crucial for enabling effective adaptive scaling, and (iii) the proposed IAS strategy reduces inference costs while maintaining final answer accuracy, utilizing less compute on more confident problems as desired. 

**Abstract (ZH)**: 过程奖励模型（PRMs）在指导大规模语言模型（LLMs）的推理时扩展算法中发挥着核心作用。然而，我们观察到即使最先进的PRMs也可能校准不佳，并且常常高估成功概率。为了解决这一问题，我们提出了一种通过分位数回归实现的校准方法，以调整PRM输出，使其更好地与真正的成功概率相匹配。利用这些校准后成功概率估计及其相关的置信区间，我们引入了一种实例自适应扩展（IAS）框架，该框架根据估计的可能产生正确最终答案的部分推理轨迹的概率动态调整推理预算。与传统方法分配固定数量的推理轨迹每个查询不同，这种方法能够在使用我们校准的PRMs时成功地适应每个实例和推理步骤。在数学推理基准测试上的实验显示，（i）我们的PRM校准方法成功地实现了小校准误差，并且优于基线方法，（ii）校准对于启用有效的自适应扩展至关重要，（iii）提出的IAS策略减少了推理成本同时保持最终答案的准确性，在更自信的问题上使用较少计算资源。 

---
# Intelligent System of Emergent Knowledge: A Coordination Fabric for Billions of Minds 

**Title (ZH)**: emergent知识智能系统：数十亿心灵的协调织 fabric 

**Authors**: Moshi Wei, Sparks Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09335)  

**Abstract**: The Intelligent System of Emergent Knowledge (ISEK) establishes a decentralized network where human and artificial intelligence agents collaborate as peers, forming a self-organizing cognitive ecosystem. Built on Web3 infrastructure, ISEK combines three fundamental principles: (1) a decentralized multi-agent architecture resistant to censorship, (2) symbiotic AI-human collaboration with equal participation rights, and (3) resilient self-adaptation through distributed consensus mechanisms.
The system implements an innovative coordination protocol featuring a six-phase workflow (Publish, Discover, Recruit, Execute, Settle, Feedback) for dynamic task allocation, supported by robust fault tolerance and a multidimensional reputation system. Economic incentives are governed by the native $ISEK token, facilitating micropayments, governance participation, and reputation tracking, while agent sovereignty is maintained through NFT-based identity management.
This synthesis of blockchain technology, artificial intelligence, and incentive engineering creates an infrastructure that actively facilitates emergent intelligence. ISEK represents a paradigm shift from conventional platforms, enabling the organic development of large-scale, decentralized cognitive systems where autonomous agents collectively evolve beyond centralized constraints. 

**Abstract (ZH)**: 基于涌现知识的智能系统（ISEK）构建了一个去中心化的网络，其中人类和人工智能代理作为平级合作，形成一个自我组织的认知生态系统。ISEK基于Web3基础设施，融合了三项基本原则：（1）去中心化的多功能代理架构，具有抗审查性，（2）共生的人工智能与人类协作，参与者拥有平等的参与权，（3）通过分布式共识机制实现的韧性自适应。系统实现了一种创新的协调协议，包含六个工作流程（发布、发现、招募、执行、结算、反馈），并具有强大的容错性和多维度声誉系统。经济激励由原生ISEK代币管理，支持微支付、治理参与和声誉追踪，代理主权则通过基于NFT的身份管理得以维护。这一区块链技术、人工智能与激励工程的综合应用，构建了一种促进涌现智能的动力结构。ISEK代表了从传统平台向一种新型范式的转变，使大规模、去中心化的认知系统能够有机发展，从而超越中心化的限制。 

---
# Multi-Agent Language Models: Advancing Cooperation, Coordination, and Adaptation 

**Title (ZH)**: 多Agent语言模型：推动合作、协调与适应 

**Authors**: Arjun Vaithilingam Sudhakar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09331)  

**Abstract**: Modern Large Language Models (LLMs) exhibit impressive zero-shot and few-shot generalization capabilities across complex natural language tasks, enabling their widespread use as virtual assistants for diverse applications such as translation and summarization. Despite being trained solely on large corpora of text without explicit supervision on author intent, LLMs appear to infer the underlying meaning of textual interactions. This raises a fundamental question: can LLMs model and reason about the intentions of others, i.e., do they possess a form of theory of mind? Understanding other's intentions is crucial for effective collaboration, which underpins human societal success and is essential for cooperative interactions among multiple agents, including humans and autonomous systems. In this work, we investigate the theory of mind in LLMs through the lens of cooperative multi-agent reinforcement learning (MARL), where agents learn to collaborate via repeated interactions, mirroring human social reasoning. Our approach aims to enhance artificial agent's ability to adapt and cooperate with both artificial and human partners. By leveraging LLM-based agents capable of natural language interaction, we move towards creating hybrid human-AI systems that can foster seamless collaboration, with broad implications for the future of human-artificial interaction. 

**Abstract (ZH)**: 现代大型语言模型（LLMs）在复杂自然语言任务中展示了卓越的零样本和少样本泛化能力，使其广泛应用于诸如翻译和总结等多种虚拟助手应用。尽管LLMs仅在大规模文本语料上进行训练，而不是在作者意图上进行显式的监督，它们似乎能够推断文本交互的潜在含义。这引发了基础性的问题：LLMs能否建模并推理他人的意图，即它们是否拥有某种形式的理论心智？理解他人的意图对于有效的协作至关重要，这支撑着人类社会的成功，并且对于多个代理（包括人类和自主系统）之间的合作互动而言是必不可少的。在本文中，我们通过合作多智能体强化学习（MARL）的视角研究LLMs的理论心智，其中智能体通过反复互动学习合作，模仿人类的社会推理过程。我们的方法旨在增强人工代理适应并与人工和人类伙伴合作的能力。借助基于LLM的能够进行自然语言交互的智能体，我们朝着创建促进无缝合作的人机混合系统迈进，对未来的这种人机交互具有广泛影响。 

---
# Alzheimer's Dementia Detection Using Perplexity from Paired Large Language Models 

**Title (ZH)**: 使用配对大型语言模型的困惑度进行阿尔茨海默病检测 

**Authors**: Yao Xiao, Heidi Christensen, Stefan Goetze  

**Link**: [PDF](https://arxiv.org/pdf/2506.09315)  

**Abstract**: Alzheimer's dementia (AD) is a neurodegenerative disorder with cognitive decline that commonly impacts language ability. This work extends the paired perplexity approach to detecting AD by using a recent large language model (LLM), the instruction-following version of Mistral-7B. We improve accuracy by an average of 3.33% over the best current paired perplexity method and by 6.35% over the top-ranked method from the ADReSS 2020 challenge benchmark. Our further analysis demonstrates that the proposed approach can effectively detect AD with a clear and interpretable decision boundary in contrast to other methods that suffer from opaque decision-making processes. Finally, by prompting the fine-tuned LLMs and comparing the model-generated responses to human responses, we illustrate that the LLMs have learned the special language patterns of AD speakers, which opens up possibilities for novel methods of model interpretation and data augmentation. 

**Abstract (ZH)**: 阿尔茨海默病痴呆（AD）是一种伴有认知下降的神经退行性疾病，常见于影响语言能力。本工作通过使用近期的大语言模型Mistral-7B的指令遵循版本，扩展了配对困惑度方法以检测AD，提高了3.33%的准确性，相对于当前最佳配对困惑度方法提高了6.35%的准确性，且优于ADReSS 2020挑战基准中的排名方法。进一步分析表明，提出的方法能够有效地检测AD，并且具有清晰可解释的决策边界，与其他具有不透明决策过程的方法形成对比。最后，通过对微调后的语言模型进行提示，并将模型生成的响应与人类响应进行比较，我们展示了语言模型学会了AD患者特有的语言模式，这为新的模型解释方法和数据增强提供了可能性。 

---
# $(RSA)^2$: A Rhetorical-Strategy-Aware Rational Speech Act Framework for Figurative Language Understanding 

**Title (ZH)**: $(RSA)^2$: 一种考虑修辞策略的语用推理框架用于描绘性语言理解 

**Authors**: Cesare Spinoso-Di Piano, David Austin, Pablo Piantanida, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.09301)  

**Abstract**: Figurative language (e.g., irony, hyperbole, understatement) is ubiquitous in human communication, resulting in utterances where the literal and the intended meanings do not match. The Rational Speech Act (RSA) framework, which explicitly models speaker intentions, is the most widespread theory of probabilistic pragmatics, but existing implementations are either unable to account for figurative expressions or require modeling the implicit motivations for using figurative language (e.g., to express joy or annoyance) in a setting-specific way. In this paper, we introduce the Rhetorical-Strategy-Aware RSA $(RSA)^2$ framework which models figurative language use by considering a speaker's employed rhetorical strategy. We show that $(RSA)^2$ enables human-compatible interpretations of non-literal utterances without modeling a speaker's motivations for being non-literal. Combined with LLMs, it achieves state-of-the-art performance on the ironic split of PragMega+, a new irony interpretation dataset introduced in this study. 

**Abstract (ZH)**: 具修辞策略意识的理性演讲行为框架 $(RSA)^2$：通过考虑说话人采用的修辞策略建模隐喻语言的使用 

---
# Causal Graph Recovery in Neuroimaging through Answer Set Programming 

**Title (ZH)**: 神经影像中基于回答集编程的因果图恢复 

**Authors**: Mohammadsajad Abavisani, Kseniya Solovyeva, David Danks, Vince Calhoun, Sergey Plis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09286)  

**Abstract**: Learning graphical causal structures from time series data presents significant challenges, especially when the measurement frequency does not match the causal timescale of the system. This often leads to a set of equally possible underlying causal graphs due to information loss from sub-sampling (i.e., not observing all possible states of the system throughout time). Our research addresses this challenge by incorporating the effects of sub-sampling in the derivation of causal graphs, resulting in more accurate and intuitive outcomes. We use a constraint optimization approach, specifically answer set programming (ASP), to find the optimal set of answers. ASP not only identifies the most probable underlying graph, but also provides an equivalence class of possible graphs for expert selection. In addition, using ASP allows us to leverage graph theory to further prune the set of possible solutions, yielding a smaller, more accurate answer set significantly faster than traditional approaches. We validate our approach on both simulated data and empirical structural brain connectivity, and demonstrate its superiority over established methods in these experiments. We further show how our method can be used as a meta-approach on top of established methods to obtain, on average, 12% improvement in F1 score. In addition, we achieved state of the art results in terms of precision and recall of reconstructing causal graph from sub-sampled time series data. Finally, our method shows robustness to varying degrees of sub-sampling on realistic simulations, whereas other methods perform worse for higher rates of sub-sampling. 

**Abstract (ZH)**: 从时间序列数据中学习图形因果结构面临显著挑战，尤其是当测量频率不能匹配系统因果时间尺度时。这常常导致由于子采样造成的的信息丢失（即，没有在整个时间范围内观察到系统的所有可能状态），使得潜在因果图有多种等可能性。我们的研究通过在因果图的推导中纳入子采样的影响，从而实现了更准确和直观的结果。我们使用约束优化方法，具体来说是回答集编程（ASP），来找到最优的答案集。ASP不仅识别出最可能的潜在图，还为专家选择提供了可能图的等价类。此外，使用ASP使得我们可以利用图论进一步修剪可能解的集合，从而比传统方法更快地获得更小、更准确的答案集。我们在模拟数据和实证的结构脑连接中验证了这一方法，并在这些实验中证明了其在F1分数上的优越性。此外，我们在从子采样时间序列数据重建因果图方面的精确度和召回率上达到了最先进的结果。最后，我们的方法在现实模拟中对不同程度的子采样表现出稳健性，而其他方法在更高的子采样率下表现较差。 

---
# UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation 

**Title (ZH)**: UAD：无监督能力蒸馏在机器人操纵中的泛化 

**Authors**: Yihe Tang, Wenlong Huang, Yingke Wang, Chengshu Li, Roy Yuan, Ruohan Zhang, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.09284)  

**Abstract**: Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD (Unsupervised Affordance Distillation), a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed $<$instruction, visual affordance$>$ pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations. Project website: this https URL 

**Abstract (ZH)**: 理解细粒度对象 affordance 对机器人在未结构化环境中根据开放式任务指令操作对象至关重要。然而，现有的视觉 affordance 预测方法往往依赖于手动标注的数据或只针对预定义的任务集。我们提出了一种 UAD（无监督 affordance 提炼）方法，该方法可以在不需要任何手动标注的情况下，从基础模型中提炼 affordance 知识并转化为任务条件下的 affordance 模型。通过利用大型视觉模型和视觉语言模型的互补优势，UAD 自动标注了一个大规模的数据集，包含详细的 $<$指令，视觉 affordance$>$ 对。仅通过冻结特征并训练一个轻量级的任务条件解码器，UAD 在野生机器人场景和各种人类活动中展现出显著的泛化能力，尽管它是基于仿真中渲染的物体进行训练的。使用 UAD 提供的 affordance 作为观察空间，我们展示了模仿学习策略，在仅进行少量（如10个）演示后，该策略能够很好地泛化到未见过的对象实例、对象类别，甚至任务指令的变化。项目网站: 这个 https URL。 

---
# Learning The Minimum Action Distance 

**Title (ZH)**: 学习最小动作距离 

**Authors**: Lorenzo Steccanella, Joshua B. Evans, Özgür Şimşek, Anders Jonsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.09276)  

**Abstract**: This paper presents a state representation framework for Markov decision processes (MDPs) that can be learned solely from state trajectories, requiring neither reward signals nor the actions executed by the agent. We propose learning the minimum action distance (MAD), defined as the minimum number of actions required to transition between states, as a fundamental metric that captures the underlying structure of an environment. MAD naturally enables critical downstream tasks such as goal-conditioned reinforcement learning and reward shaping by providing a dense, geometrically meaningful measure of progress. Our self-supervised learning approach constructs an embedding space where the distances between embedded state pairs correspond to their MAD, accommodating both symmetric and asymmetric approximations. We evaluate the framework on a comprehensive suite of environments with known MAD values, encompassing both deterministic and stochastic dynamics, as well as discrete and continuous state spaces, and environments with noisy observations. Empirical results demonstrate that the proposed approach not only efficiently learns accurate MAD representations across these diverse settings but also significantly outperforms existing state representation methods in terms of representation quality. 

**Abstract (ZH)**: 本文提出了一种仅从状态轨迹学习马尔可夫决策过程（MDPs）状态表示的框架，无需奖励信号或智能体执行的动作。我们提出学习最小动作距离（MAD），定义为在两个状态之间进行转换所需的最小动作数量，作为捕获环境潜在结构的基本度量。MAD 自然地支持目标条件强化学习和奖励塑形等关键下游任务，提供了一个密集的、几何意义上具有意义的进步度量。我们提出了一种自监督学习方法，构建了嵌入空间，其中嵌入状态对之间的距离对应于它们的MAD，既支持对称近似也支持非对称近似。我们在包含已知MAD值的广泛环境套件上进行了评估，这些环境涵盖了确定性和随机动力学，以及离散和连续状态空间，还包括具有噪声观测值的环境。实验证明，所提出的方法不仅在这些多样化的环境中高效地学习了准确的MAD表示，而且在表示质量上显著优于现有状态表示方法。 

---
# A Multi-Armed Bandit Framework for Online Optimisation in Green Integrated Terrestrial and Non-Terrestrial Networks 

**Title (ZH)**: 基于多臂bandit框架的绿色集成 terrestrial 和非terrestrial 网络的在线优化 

**Authors**: Henri Alam, Antonio de Domenico, Tareq Si Salem, Florian Kaltenberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.09268)  

**Abstract**: Integrated terrestrial and non-terrestrial network (TN-NTN) architectures offer a promising solution for expanding coverage and improving capacity for the network. While non-terrestrial networks (NTNs) are primarily exploited for these specific reasons, their role in alleviating terrestrial network (TN) load and enabling energy-efficient operation has received comparatively less attention. In light of growing concerns associated with the densification of terrestrial deployments, this work aims to explore the potential of NTNs in supporting a more sustainable network. In this paper, we propose a novel online optimisation framework for integrated TN-NTN architectures, built on a multi-armed bandit (MAB) formulation and leveraging the Bandit-feedback Constrained Online Mirror Descent (BCOMD) algorithm. Our approach adaptively optimises key system parameters--including bandwidth allocation, user equipment (UE) association, and macro base station (MBS) shutdown--to balance network capacity and energy efficiency in real time. Extensive system-level simulations over a 24-hour period show that our framework significantly reduces the proportion of unsatisfied UEs during peak hours and achieves up to 19% throughput gains and 5% energy savings in low-traffic periods, outperforming standard network settings following 3GPP recommendations. 

**Abstract (ZH)**: 集成陆地和非陆地网络(TN-NTN)架构提供了扩展覆盖范围和提高网络容量的有希望的解决方案。尽管非陆地网络(NTNs)主要用于这些特定目的，但它们在缓解陆地网络(TNs)负载和实现节能操作方面的角色受到了较少关注。鉴于对陆地部署密集化的担忧不断增加，本工作旨在探讨NTNs在支持可持续网络方面的潜力。在本文中，我们提出了一种基于多臂 bandit (MAB) 表述并利用 Bandit-feedback Constrained Online Mirror Descent (BCOMD) 算法的新型在线优化框架，以实现集成TN-NTN架构，我们的方法实现实时平衡网络容量和节能。在24小时的系统级仿真中，我们的框架在高峰时段显著减少了未满足用户设备的比例，并在低流量时段实现了高达19%的吞吐量增益和5%的节能效果，优于遵循3GPP建议的标准网络设置。 

---
# Self-Anchored Attention Model for Sample-Efficient Classification of Prosocial Text Chat 

**Title (ZH)**: 基于自我锚定注意力模型的高效分类亲社会文本聊天 

**Authors**: Zhuofang Li, Rafal Kocielnik, Fereshteh Soltani, Penphob, Boonyarungsrit, Animashree Anandkumar, R. Michael Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2506.09259)  

**Abstract**: Millions of players engage daily in competitive online games, communicating through in-game chat. Prior research has focused on detecting relatively small volumes of toxic content using various Natural Language Processing (NLP) techniques for the purpose of moderation. However, recent studies emphasize the importance of detecting prosocial communication, which can be as crucial as identifying toxic interactions. Recognizing prosocial behavior allows for its analysis, rewarding, and promotion. Unlike toxicity, there are limited datasets, models, and resources for identifying prosocial behaviors in game-chat text. In this work, we employed unsupervised discovery combined with game domain expert collaboration to identify and categorize prosocial player behaviors from game chat. We further propose a novel Self-Anchored Attention Model (SAAM) which gives 7.9% improvement compared to the best existing technique. The approach utilizes the entire training set as "anchors" to help improve model performance under the scarcity of training data. This approach led to the development of the first automated system for classifying prosocial behaviors in in-game chats, particularly given the low-resource settings where large-scale labeled data is not available. Our methodology was applied to one of the most popular online gaming titles - Call of Duty(R): Modern Warfare(R)II, showcasing its effectiveness. This research is novel in applying NLP techniques to discover and classify prosocial behaviors in player in-game chat communication. It can help shift the focus of moderation from solely penalizing toxicity to actively encouraging positive interactions on online platforms. 

**Abstract (ZH)**: 大规模游戏玩家每日参与竞争性在线游戏，并通过游戏内聊天进行交流。先前的研究专注于使用各种自然语言处理（NLP）技术检测相对少量的有毒内容以进行管理。然而，近期的研究强调了检测亲社会交流的重要性，这种交流与识别有毒互动同样重要。识别亲社会行为使其能够被分析、奖励和推广。与毒性识别相比，游戏聊天文本中识别亲社会行为的数据集、模型和资源有限。在本工作中，我们结合无监督发现与游戏领域专家合作，从游戏聊天中识别和分类玩家的亲社会行为。进一步提出了一种新颖的自我锚定注意力模型（SAAM），相比现有最佳技术提高了7.9%。该方法利用整个训练集作为“锚点”来帮助改善在数据稀缺情况下的模型性能。该方法开发了第一个自动化的系统用于分类游戏内聊天中的亲社会行为，特别是在大规模标注数据不可用的低资源环境下。该方法被应用于最受欢迎的在线游戏之一——使命召唤®：现代战争®II，展示了其效果。这项研究是将NLP技术应用于发现和分类玩家游戏内聊天中的亲社会行为的先驱工作，有助于将管理的重点从仅仅惩罚毒性转向积极鼓励在线平台上的正面互动。 

---
# Extrapolation by Association: Length Generalization Transfer in Transformers 

**Title (ZH)**: 关联外推：Transformer中的长度泛化迁移学习 

**Authors**: Ziyang Cai, Nayoung Lee, Avi Schwarzschild, Samet Oymak, Dimitris Papailiopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.09251)  

**Abstract**: Transformer language models have demonstrated impressive generalization capabilities in natural language domains, yet we lack a fine-grained understanding of how such generalization arises. In this paper, we investigate length generalization--the ability to extrapolate from shorter to longer inputs--through the lens of \textit{task association}. We find that length generalization can be \textit{transferred} across related tasks. That is, training a model with a longer and related auxiliary task can lead it to generalize to unseen and longer inputs from some other target task. We demonstrate this length generalization transfer across diverse algorithmic tasks, including arithmetic operations, string transformations, and maze navigation. Our results show that transformer models can inherit generalization capabilities from similar tasks when trained jointly. Moreover, we observe similar transfer effects in pretrained language models, suggesting that pretraining equips models with reusable computational scaffolding that facilitates extrapolation in downstream settings. Finally, we provide initial mechanistic evidence that length generalization transfer correlates with the re-use of the same attention heads between the tasks. Together, our findings deepen our understanding of how transformers generalize to out-of-distribution inputs and highlight the compositional reuse of inductive structure across tasks. 

**Abstract (ZH)**: 基于任务关联的Transformer语言模型的长度泛化迁移研究 

---
# Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs 

**Title (ZH)**: 自适应排序变换器输出的稳健噪声抑制 

**Authors**: Greyson Brothers  

**Link**: [PDF](https://arxiv.org/pdf/2506.09215)  

**Abstract**: We investigate the design of pooling methods used to summarize the outputs of transformer embedding models, primarily motivated by reinforcement learning and vision applications. This work considers problems where a subset of the input vectors contains requisite information for a downstream task (signal) while the rest are distractors (noise). By framing pooling as vector quantization with the goal of minimizing signal loss, we demonstrate that the standard methods used to aggregate transformer outputs, AvgPool, MaxPool, and ClsToken, are vulnerable to performance collapse as the signal-to-noise ratio (SNR) of inputs fluctuates. We then show that an attention-based adaptive pooling method can approximate the signal-optimal vector quantizer within derived error bounds for any SNR. Our theoretical results are first validated by supervised experiments on a synthetic dataset designed to isolate the SNR problem, then generalized to standard relational reasoning, multi-agent reinforcement learning, and vision benchmarks with noisy observations, where transformers with adaptive pooling display superior robustness across tasks. 

**Abstract (ZH)**: 我们考察了用于总结变压器嵌入模型输出的池化方法的设计，主要受强化学习和视觉应用的启发。本文考虑了一种情况，即输入向量中的子集包含了下游任务所需的信息（信号），而其余的则为干扰信息（噪声）。通过将池化视为向量量化，并以最小化信号丢失为目 标，我们证明了标准的变压器输出聚合方法，如 AvgPool、MaxPool 和 ClsToken，在输入信噪比（SNR）波动时容易导致性能崩溃。然后我们展示了基于注意力的自适应池化方法可以在任何信噪比下近似信号优化的向量量化器。我们的理论结果首先通过在合成数据集上进行监督实验进行验证，该数据集旨在孤立信噪比问题，随后泛化到标准关系推理、多智能体强化学习和具有嘈杂观察的视觉基准任务，其中使用自适应池化的方法在各种任务上表现出更强的鲁棒性。 

---
# SimClass: A Classroom Speech Dataset Generated via Game Engine Simulation For Automatic Speech Recognition Research 

**Title (ZH)**: SimClass: 通过游戏引擎模拟生成的课堂语音数据集及其在自动语音识别研究中的应用 

**Authors**: Ahmed Adel Attia, Jing Liu, Carl Espy-Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2506.09206)  

**Abstract**: The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Public classroom datasets remain limited, and the lack of a dedicated classroom noise corpus prevents the use of standard data augmentation techniques.
In this paper, we introduce a scalable methodology for synthesizing classroom noise using game engines, a framework that extends to other domains. Using this methodology, we present SimClass, a dataset that includes both a synthesized classroom noise corpus and a simulated classroom speech dataset. The speech data is generated by pairing a public children's speech corpus with YouTube lecture videos to approximate real classroom interactions in clean conditions. Our experiments on clean and noisy speech demonstrate that SimClass closely approximates real classroom speech, making it a valuable resource for developing robust speech recognition and enhancement models. 

**Abstract (ZH)**: 大规模课堂语音数据的稀缺性阻碍了基于AI的教育语音模型的发展。公开的课堂数据集仍然有限，缺乏专门的课堂噪声语料库使得标准数据增强技术无法使用。
在本文中，我们介绍了一种使用游戏引擎合成课堂噪声的可扩展方法，该框架适用于其他领域。利用这种方法，我们呈现了SimClass数据集，该数据集包括合成课堂噪声语料库和模拟课堂语音数据集。语音数据是通过将一个公开的儿童语音语料库与YouTube讲座视频配对生成的，以在干净条件下逼近真实课堂互动。我们的实验证明，在干净和嘈杂的语音上，SimClass 都能逼近真实课堂语音，使其成为开发稳健的语音识别和增强模型的重要资源。 

---
# A Topological Improvement of the Overall Performance of Sparse Evolutionary Training: Motif-Based Structural Optimization of Sparse MLPs Project 

**Title (ZH)**: 基于动力学优化的稀疏演化训练整体性能拓扑改进：稀疏MLP结构的模式基优化 

**Authors**: Xiaotian Chen, Hongyun Liu, Seyed Sahand Mohammadi Ziabari  

**Link**: [PDF](https://arxiv.org/pdf/2506.09204)  

**Abstract**: Deep Neural Networks (DNNs) have been proven to be exceptionally effective and have been applied across diverse domains within deep learning. However, as DNN models increase in complexity, the demand for reduced computational costs and memory overheads has become increasingly urgent. Sparsity has emerged as a leading approach in this area. The robustness of sparse Multi-layer Perceptrons (MLPs) for supervised feature selection, along with the application of Sparse Evolutionary Training (SET), illustrates the feasibility of reducing computational costs without compromising accuracy. Moreover, it is believed that the SET algorithm can still be improved through a structural optimization method called motif-based optimization, with potential efficiency gains exceeding 40% and a performance decline of under 4%. This research investigates whether the structural optimization of Sparse Evolutionary Training applied to Multi-layer Perceptrons (SET-MLP) can enhance performance and to what extent this improvement can be achieved. 

**Abstract (ZH)**: 深神经网络（DNNs）已被证明在深度学习的各个领域内表现出色且应用广泛。然而，随着DNN模型复杂性的增加，降低计算成本和内存开销的需求愈发迫切。稀疏性已成为这一领域的主导方法之一。基于监督特征选择的稀疏多层感知机（MLPs）的稳健性以及稀疏进化训练（SET）的应用表明，在不牺牲准确性的前提下削减计算成本的可能性。此外，通过一种称为图样基于优化的结构优化方法，SET算法仍有改进空间，预期效率提升可达40%以上，性能下降不超过4%。本研究旨在探讨将结构优化应用于多层感知机的稀疏进化训练（SET-MLP）是否能提高性能，以及这种改进的程度。 

---
# Policy-Based Trajectory Clustering in Offline Reinforcement Learning 

**Title (ZH)**: 基于策略的轨迹聚类在离线强化学习中 

**Authors**: Hao Hu, Xinqi Wang, Simon Shaolei Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.09202)  

**Abstract**: We introduce a novel task of clustering trajectories from offline reinforcement learning (RL) datasets, where each cluster center represents the policy that generated its trajectories. By leveraging the connection between the KL-divergence of offline trajectory distributions and a mixture of policy-induced distributions, we formulate a natural clustering objective. To solve this, we propose Policy-Guided K-means (PG-Kmeans) and Centroid-Attracted Autoencoder (CAAE). PG-Kmeans iteratively trains behavior cloning (BC) policies and assigns trajectories based on policy generation probabilities, while CAAE resembles the VQ-VAE framework by guiding the latent representations of trajectories toward the vicinity of specific codebook entries to achieve clustering. Theoretically, we prove the finite-step convergence of PG-Kmeans and identify a key challenge in offline trajectory clustering: the inherent ambiguity of optimal solutions due to policy-induced conflicts, which can result in multiple equally valid but structurally distinct clusterings. Experimentally, we validate our methods on the widely used D4RL dataset and custom GridWorld environments. Our results show that both PG-Kmeans and CAAE effectively partition trajectories into meaningful clusters. They offer a promising framework for policy-based trajectory clustering, with broad applications in offline RL and beyond. 

**Abstract (ZH)**: 一种新的离线强化学习轨迹聚类任务及其方法：政策引导的K-means（PG-Kmeans）和中心吸引自编码器（CAAE） 

---
# FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models 

**Title (ZH)**: FLoRIST: 特殊值阈值化方法实现高效准确的联邦微调大语言模型 

**Authors**: Hariharan Ramesh, Jyotikrishna Dass  

**Link**: [PDF](https://arxiv.org/pdf/2506.09199)  

**Abstract**: Integrating Low-Rank Adaptation (LoRA) into federated learning offers a promising solution for parameter-efficient fine-tuning of Large Language Models (LLMs) without sharing local data. However, several methods designed for federated LoRA present significant challenges in balancing communication efficiency, model accuracy, and computational cost, particularly among heterogeneous clients. These methods either rely on simplistic averaging of local adapters, which introduces aggregation noise, require transmitting large stacked local adapters, leading to poor communication efficiency, or necessitate reconstructing memory-dense global weight-update matrix and performing computationally expensive decomposition to design client-specific low-rank adapters. In this work, we propose FLoRIST, a federated fine-tuning framework that achieves mathematically accurate aggregation without incurring high communication or computational overhead. Instead of constructing the full global weight-update matrix at the server, FLoRIST employs an efficient decomposition pipeline by performing singular value decomposition on stacked local adapters separately. This approach operates within a compact intermediate space to represent the accumulated information from local LoRAs. We introduce tunable singular value thresholding for server-side optimal rank selection to construct a pair of global low-rank adapters shared by all clients. Extensive empirical evaluations across multiple datasets and LLMs demonstrate that FLoRIST consistently strikes the best balance between superior communication efficiency and competitive performance in both homogeneous and heterogeneous setups. 

**Abstract (ZH)**: 将低秩适应（LoRA）集成到联邦学习中，为大规模语言模型（LLMs）的参数高效微调提供了有望的解决方案，无需共享本地数据。然而，为联邦LoRA设计的几种方法在平衡通信效率、模型准确性和计算成本方面面临着显著挑战，尤其是在异构客户端之间。这些方法要么依赖于简单的本地适配器平均，引入了聚合噪声，要么需要传输大型堆叠的本地适配器，导致较差的通信效率，要么需要重建密集的记忆权重更新矩阵，并进行计算昂贵的分解来设计客户端特定的低秩适配器。在本文中，我们提出了一种联邦微调框架FLoRIST，该框架在不引入高通信或计算开销的情况下实现了数学上准确的聚合。FLoRIST 服务器不构建完整的全局权重更新矩阵，而是通过分别对堆叠的本地适配器进行奇异值分解来使用高效的分解管道。该方法在紧凑的中间空间中表示来自本地LoRA的累积信息。我们引入了可调的奇异值阈值优化服务器端的秩选择，构建由所有客户端共享的一对全局低秩适配器。在多个数据集和大语言模型上的广泛实证评估表明，FLoRIST 在同构和异构设置中始终能够实现通信效率和性能的最优平衡。 

---
# Graph Attention-based Decentralized Actor-Critic for Dual-Objective Control of Multi-UAV Swarms 

**Title (ZH)**: 基于图注意力的去中心化演员-评论家方法及其在多无人机群双目标控制中的应用 

**Authors**: Haoran Peng, Ying-Jun Angela Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09195)  

**Abstract**: This research focuses on optimizing multi-UAV systems with dual objectives: maximizing service coverage as the primary goal while extending battery lifetime as the secondary objective. We propose a Graph Attention-based Decentralized Actor-Critic (GADC) to optimize the dual objectives. The proposed approach leverages a graph attention network to process UAVs' limited local observation and reduce the dimension of the environment states. Subsequently, an actor-double-critic network is developed to manage dual policies for joint objective optimization. The proposed GADC uses a Kullback-Leibler (KL) divergence factor to balance the tradeoff between coverage performance and battery lifetime in the multi-UAV system. We assess the scalability and efficiency of GADC through comprehensive benchmarking against state-of-the-art methods, considering both theory and experimental aspects. Extensive testing in both ideal settings and NVIDIA Sionna's realistic ray tracing environment demonstrates GADC's superior performance. 

**Abstract (ZH)**: 基于图注意力的解耦actor-critic方法优化多无人机系统的双重目标：最大化服务覆盖与延长电池寿命 

---
# Integration of Contrastive Predictive Coding and Spiking Neural Networks 

**Title (ZH)**: 对比预测编码与尖峰神经网络的集成 

**Authors**: Emirhan Bilgiç, Neslihan Serap Şengör, Namık Berk Yalabık, Yavuz Selim İşler, Aykut Görkem Gelen, Rahmi Elibol  

**Link**: [PDF](https://arxiv.org/pdf/2506.09194)  

**Abstract**: This study examines the integration of Contrastive Predictive Coding (CPC) with Spiking Neural Networks (SNN). While CPC learns the predictive structure of data to generate meaningful representations, SNN mimics the computational processes of biological neural systems over time. In this study, the goal is to develop a predictive coding model with greater biological plausibility by processing inputs and outputs in a spike-based system. The proposed model was tested on the MNIST dataset and achieved a high classification rate in distinguishing positive sequential samples from non-sequential negative samples. The study demonstrates that CPC can be effectively combined with SNN, showing that an SNN trained for classification tasks can also function as an encoding mechanism. Project codes and detailed results can be accessed on our GitHub page: this https URL 

**Abstract (ZH)**: 本研究考察了对比预测编码（CPC）与尖峰神经网络（SNN）的结合。在CPC学习数据的预测结构以生成有意义的表示的同时，SNN模拟了生物神经系统的计算过程。本研究的目标是通过基于尖峰的系统处理输入和输出来开发具有更强生物合理性的预测编码模型。所提出的方法在MNIST数据集上进行了测试，并实现了高分类率，能够区分正序列样本和非序列负样本。研究表明，CPC可以有效地结合到SNN中，表明SNN在训练分类任务时也可以作为编码机制。项目代码和详细结果可访问我们的GitHub页面：this https URL 

---
# Multi-Task Reward Learning from Human Ratings 

**Title (ZH)**: 多任务奖励学习从人类评价 

**Authors**: Mingkang Wu, Devin White, Evelyn Rose, Vernon Lawhern, Nicholas R Waytowich, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09183)  

**Abstract**: Reinforcement learning from human feeback (RLHF) has become a key factor in aligning model behavior with users' goals. However, while humans integrate multiple strategies when making decisions, current RLHF approaches often simplify this process by modeling human reasoning through isolated tasks such as classification or regression. In this paper, we propose a novel reinforcement learning (RL) method that mimics human decision-making by jointly considering multiple tasks. Specifically, we leverage human ratings in reward-free environments to infer a reward function, introducing learnable weights that balance the contributions of both classification and regression models. This design captures the inherent uncertainty in human decision-making and allows the model to adaptively emphasize different strategies. We conduct several experiments using synthetic human ratings to validate the effectiveness of the proposed approach. Results show that our method consistently outperforms existing rating-based RL methods, and in some cases, even surpasses traditional RL approaches. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）已成为使模型行为与用户目标相一致的关键因素。然而，尽管人类在做决策时会整合多种策略，当前的RLHF方法往往通过分类或回归等孤立任务简化这一过程来建模人类推理。在本文中，我们提出了一种新的强化学习（RL）方法，通过同时考虑多个任务来模仿人类的决策过程。具体而言，我们利用奖励免费环境中的人类评分来推断奖励函数，并引入可学习的权重来平衡分类和回归模型的贡献。这种设计捕捉了人类决策过程中的固有不确定性，并允许模型适应性地强调不同的策略。我们使用合成的人类评分进行了多项实验，以验证所提出方法的有效性。结果表明，我们的方法在多个方面都优于现有基于评分的RL方法，并在某些情况下甚至超越了传统RL方法。 

---
# PHRASED: Phrase Dictionary Biasing for Speech Translation 

**Title (ZH)**: 短语字典偏向：用于语音Translate的短语字典偏向 

**Authors**: Peidong Wang, Jian Xue, Rui Zhao, Junkun Chen, Aswin Shanmugam Subramanian, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09175)  

**Abstract**: Phrases are essential to understand the core concepts in conversations. However, due to their rare occurrence in training data, correct translation of phrases is challenging in speech translation tasks. In this paper, we propose a phrase dictionary biasing method to leverage pairs of phrases mapping from the source language to the target language. We apply the phrase dictionary biasing method to two types of widely adopted models, a transducer-based streaming speech translation model and a multimodal large language model. Experimental results show that the phrase dictionary biasing method outperforms phrase list biasing by 21% relatively for the streaming speech translation model. In addition, phrase dictionary biasing enables multimodal large language models to use external phrase information, achieving 85% relative improvement in phrase recall. 

**Abstract (ZH)**: 短语在理解对话的核心概念中至关重要。然而，由于短语在训练数据中罕见出现，其在语音翻译任务中的正确翻译具有挑战性。本文提出了一种短语字典偏置方法，利用源语言到目标语言的短语映射对。我们将短语字典偏置方法应用于两种广泛采用的模型：基于转换器的流式语音翻译模型和多模态大型语言模型。实验结果表明，短语字典偏置方法在流式语音翻译模型中的相对表现优于短语列表偏置21%。此外，短语字典偏置使多模态大型语言模型能够利用外部短语信息，实现了短语召回率85%的相对提升。 

---
# Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search 

**Title (ZH)**: 基于原子事实增强和前瞻搜索的上下文学习以提升LLM代理规划能力 

**Authors**: Samuel Holt, Max Ruiz Luyten, Thomas Pouplin, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09171)  

**Abstract**: Large Language Models (LLMs) are increasingly capable but often require significant guidance or extensive interaction history to perform effectively in complex, interactive environments. Existing methods may struggle with adapting to new information or efficiently utilizing past experiences for multi-step reasoning without fine-tuning. We introduce a novel LLM agent framework that enhances planning capabilities through in-context learning, facilitated by atomic fact augmentation and a recursive lookahead search. Our agent learns to extract task-critical ``atomic facts'' from its interaction trajectories. These facts dynamically augment the prompts provided to LLM-based components responsible for action proposal, latent world model simulation, and state-value estimation. Planning is performed via a depth-limited lookahead search, where the LLM simulates potential trajectories and evaluates their outcomes, guided by the accumulated facts and interaction history. This approach allows the agent to improve its understanding and decision-making online, leveraging its experience to refine its behavior without weight updates. We provide a theoretical motivation linking performance to the quality of fact-based abstraction and LLM simulation accuracy. Empirically, our agent demonstrates improved performance and adaptability on challenging interactive tasks, achieving more optimal behavior as it accumulates experience, showcased in tasks such as TextFrozenLake and ALFWorld. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益具备强大能力，但在复杂交互环境中有效运行时常需要显著的引导或大量的交互历史。现有方法可能在适应新信息或高效利用过去经验进行多步推理时遇到困难，无需微调的情况下。我们提出了一种新颖的LLM代理框架，通过上下文学习增强规划能力，借助原子事实增强和递归前瞻搜索实现。代理能够从其交互轨迹中学习提取任务关键的“原子事实”。这些事实动态增强负责动作提议、潜在世界模型模拟和状态价值估计的LLM组件所接收的提示。规划通过深度限制的前瞻搜索进行，其中LLM模拟潜在轨迹并评估其结果，受到累积事实和交互历史的指导。该方法使代理能够在不更新权重的情况下，在线改善其理解和决策能力，利用其经验来优化行为。我们提供了将性能与基于事实的抽象质量和LLM模拟准确性联系起来的理论动机。实验中，我们的代理在挑战性的交互任务中表现出改进的表现和适应性，随着经验的积累，表现出更加优化的行为，这在诸如TextFrozenLake和ALFWorld等任务中得以展示。 

---
# Estimating Visceral Adiposity from Wrist-Worn Accelerometry 

**Title (ZH)**: 从腕戴加速度计估计内脏脂肪量 

**Authors**: James R. Williamson, Andrew Alini, Brian A. Telfer, Adam W. Potter, Karl E. Friedl  

**Link**: [PDF](https://arxiv.org/pdf/2506.09167)  

**Abstract**: Visceral adipose tissue (VAT) is a key marker of both metabolic health and habitual physical activity (PA). Excess VAT is highly correlated with type 2 diabetes and insulin resistance. The mechanistic basis for this pathophysiology relates to overloading the liver with fatty acids. VAT is also a highly labile fat depot, with increased turnover stimulated by catecholamines during exercise. VAT can be measured with sophisticated imaging technologies, but can also be inferred directly from PA. We tested this relationship using National Health and Nutrition Examination Survey (NHANES) data from 2011-2014, for individuals aged 20-60 years with 7 days of accelerometry data (n=2,456 men; 2,427 women) [1]. Two approaches were used for estimating VAT from activity. The first used engineered features based on movements during gait and sleep, and then ridge regression to map summary statistics of these features into a VAT estimate. The second approach used deep neural networks trained on 24 hours of continuous accelerometry. A foundation model first mapped each 10s frame into a high-dimensional feature vector. A transformer model then mapped each day's feature vector time series into a VAT estimate, which were averaged over multiple days. For both approaches, the most accurate estimates were obtained with the addition of covariate information about subject demographics and body measurements. The best performance was obtained by combining the two approaches, resulting in VAT estimates with correlations of r=0.86. These findings demonstrate a strong relationship between PA and VAT and, by extension, between PA and metabolic health risks. 

**Abstract (ZH)**: 内脏脂肪组织（VAT）是代谢健康和习惯性体力活动（PA）的关键标志物。过多的VAT与2型糖尿病和胰岛素抵抗高度相关。这种病理生理机制的基础在于向肝脏过量提供脂肪酸。VAT也是一种高度可变的脂肪储存部位，在运动期间由儿茶酚胺刺激其周转率的增加。VAT可以使用复杂的成像技术进行测量，但也可以通过体力活动直接推断。我们使用2011-2014年全国健康和营养 Examination Survey (NHANES) 数据（年龄在20-60岁之间，男性2,456人；女性2,427人，有7天的加速度计数据）测试了这种关系 [1]。估测VAT的两种方法均使用了体力活动。第一种方法基于行走和睡眠期间的运动工程特征，并使用岭回归将这些特征的摘要统计映射到VAT估计值。第二种方法使用了在24小时连续加速度计数据上训练的深度神经网络。基础模型将每个10秒帧映射到高维特征向量。然后，变压器模型将每个特征向量时间序列映射到VAT估计值，并在多天内进行平均。对于这两种方法，最准确的估计值是通过添加关于受试者的人口统计学和身体测量的协变量信息获得的。通过结合两种方法，性能最佳，结果的VAT估计值的相关性为r=0.86。这些发现表明，体力活动与VAT之间存在强烈关系，进而推断体力活动与代谢健康风险之间的关系。 

---
# Understanding Human-AI Trust in Education 

**Title (ZH)**: 理解教育中的人工智能信任问题 

**Authors**: Griffin Pitts, Sanaz Motamedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09160)  

**Abstract**: As AI chatbots become increasingly integrated in education, students are turning to these systems for guidance, feedback, and information. However, the anthropomorphic characteristics of these chatbots create ambiguity regarding whether students develop trust toward them as they would a human peer or instructor, based in interpersonal trust, or as they would any other piece of technology, based in technology trust. This ambiguity presents theoretical challenges, as interpersonal trust models may inappropriately ascribe human intentionality and morality to AI, while technology trust models were developed for non-social technologies, leaving their applicability to anthropomorphic systems unclear. To address this gap, we investigate how human-like and system-like trusting beliefs comparatively influence students' perceived enjoyment, trusting intention, behavioral intention to use, and perceived usefulness of an AI chatbot - factors associated with students' engagement and learning outcomes. Through partial least squares structural equation modeling, we found that human-like and system-like trust significantly influenced student perceptions, with varied effects. Human-like trust more strongly predicted trusting intention, while system-like trust better predicted behavioral intention and perceived usefulness. Both had similar effects on perceived enjoyment. Given the partial explanatory power of each type of trust, we propose that students develop a distinct form of trust with AI chatbots (human-AI trust) that differs from human-human and human-technology models of trust. Our findings highlight the need for new theoretical frameworks specific to human-AI trust and offer practical insights for fostering appropriately calibrated trust, which is critical for the effective adoption and pedagogical impact of AI in education. 

**Abstract (ZH)**: 随着AI聊天机器人在教育中的日益集成，学生开始依赖这些系统寻求指导、反馈和信息。然而，这些聊天机器人的拟人特性使得学生是更像对其人类同伴或教师产生人际信任，还是基于技术信任对其产生信任变得模糊。这种模糊性提出了理论挑战，因为人际信任模型可能会不合适地赋予AI人类意图和道德，而技术信任模型是为非社交技术开发的，其适用于拟人系统的效果尚不清楚。为了解决这一差距，我们探讨了拟人信任信念和系统信任信念在比较上如何影响学生对AI聊天机器人的感知愉悦度、信任意图、使用意图和感知有用性，这些因素与学生的参与度和学习成果相关。通过部分最小二乘结构方程建模，我们发现拟人信任和系统信任显著影响了学生的态度，但影响效果有所不同。拟人信任更强烈地预测了信任意图，而系统信任更好地预测了使用意图和感知有用性。两种信任在感知愉悦度上的影响相似。鉴于每种信任的解释力有限，我们认为学生形成了与人类-人类和人类-技术信任模型不同的对AI聊天机器人的信任形式（人类-AI信任）。我们的研究凸显了需要针对人类-AI信任的新理论框架的必要性，并提供了培养适当校准信任的实用见解，这对于AI在教育中的有效采用及其教育影响至关重要。 

---
# LLM-as-a-qualitative-judge: automating error analysis in natural language generation 

**Title (ZH)**: LLM作为定性裁判：自动化自然语言生成中的错误分析 

**Authors**: Nadezhda Chirkova, Tunde Oluwaseyi Ajayi, Seth Aycock, Zain Muhammad Mujahid, Vladana Perlić, Ekaterina Borisova, Markarit Vartampetian  

**Link**: [PDF](https://arxiv.org/pdf/2506.09147)  

**Abstract**: Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 将大型语言模型（LLMs） prompting 生成文本进行评估，即LLM-as-a-judge，已成为自然语言生成（NLG）的标准评估方法，但主要作为一种定量工具，即以数值分数为主要输出。本文提出LLM-as-a-qualitative-judge，这是一种基于LLM的评估方法，主要输出是对NLG系统输出中常见问题类型的结构化报告。该方法旨在为开发者提供有关如何改进给定NLG系统的有意义见解，包括两个主要步骤：实例问题开放分析和使用直观累加算法对发现的问题进行聚类。我们还引入了一种评估所提方法的策略，并提供了来自12个NLG数据集的约300个问题注解实例的标注。结果显示，LLM-as-a-qualitative-judge在2/3的情况下正确识别了实例特定的问题，并能够生成类似于人工注释者所编写的错误类型报告。我们的代码和数据可在以下网址公开访问：this https URL。 

---
# SensorLM: Learning the Language of Wearable Sensors 

**Title (ZH)**: SensorLM：学习可穿戴传感器的语言 

**Authors**: Yuwei Zhang, Kumar Ayush, Siyuan Qiao, A. Ali Heydari, Girish Narayanswamy, Maxwell A. Xu, Ahmed A. Metwally, Shawn Xu, Jake Garrison, Xuhai Xu, Tim Althoff, Yun Liu, Pushmeet Kohli, Jiening Zhan, Mark Malhotra, Shwetak Patel, Cecilia Mascolo, Xin Liu, Daniel McDuff, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09108)  

**Abstract**: We present SensorLM, a family of sensor-language foundation models that enable wearable sensor data understanding with natural language. Despite its pervasive nature, aligning and interpreting sensor data with language remains challenging due to the lack of paired, richly annotated sensor-text descriptions in uncurated, real-world wearable data. We introduce a hierarchical caption generation pipeline designed to capture statistical, structural, and semantic information from sensor data. This approach enabled the curation of the largest sensor-language dataset to date, comprising over 59.7 million hours of data from more than 103,000 people. Furthermore, SensorLM extends prominent multimodal pretraining architectures (e.g., CLIP, CoCa) and recovers them as specific variants within a generic architecture. Extensive experiments on real-world tasks in human activity analysis and healthcare verify the superior performance of SensorLM over state-of-the-art in zero-shot recognition, few-shot learning, and cross-modal retrieval. SensorLM also demonstrates intriguing capabilities including scaling behaviors, label efficiency, sensor captioning, and zero-shot generalization to unseen tasks. 

**Abstract (ZH)**: 我们呈现了SensorLM，这是一个传感器-语言基础模型家族，能够利用自然语言理解穿戴式传感器数据。尽管传感器数据与语言的对齐和解释具有普遍性，但由于未整理的真实世界穿戴数据中缺乏配对的、标注丰富的传感器-文本描述，这一过程仍然具有挑战性。我们引入了一个分层标题生成流水线，旨在从传感器数据中捕捉统计、结构和语义信息。这一方法促成了迄今为止最大的传感器-语言数据集的编目，包含超过103,000人的5970多万小时数据。此外，SensorLM 扩展了著名的多模态预训练架构（如 CLIP、CoCa），并将其作为通用架构下的特定变体。在人类活动分析和医疗保健等实际任务上的广泛实验验证了SensorLM 在零样本识别、少样本学习和跨模态检索方面的优越性能。SensorLM 还展示了扩展行为、标签效率、传感器标题生成和对未见任务的零样本泛化等有趣的特性。 

---
# FAIRTOPIA: Envisioning Multi-Agent Guardianship for Disrupting Unfair AI Pipelines 

**Title (ZH)**: FAIRTOPIA: 构想多agent监护以颠覆不公平AIpipeline 

**Authors**: Athena Vakali, Ilias Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09107)  

**Abstract**: AI models have become active decision makers, often acting without human supervision. The rapid advancement of AI technology has already caused harmful incidents that have hurt individuals and societies and AI unfairness in heavily criticized. It is urgent to disrupt AI pipelines which largely neglect human principles and focus on computational biases exploration at the data (pre), model(in), and deployment (post) processing stages. We claim that by exploiting the advances of agents technology, we will introduce cautious, prompt, and ongoing fairness watch schemes, under realistic, systematic, and human-centric fairness expectations. We envision agents as fairness guardians, since agents learn from their environment, adapt to new information, and solve complex problems by interacting with external tools and other systems. To set the proper fairness guardrails in the overall AI pipeline, we introduce a fairness-by-design approach which embeds multi-role agents in an end-to-end (human to AI) synergetic scheme. Our position is that we may design adaptive and realistic AI fairness frameworks, and we introduce a generalized algorithm which can be customized to the requirements and goals of each AI decision making scenario. Our proposed, so called FAIRTOPIA framework, is structured over a three-layered architecture, which encapsulates the AI pipeline inside an agentic guardian and a knowledge-based, self-refining layered scheme. Based on our proposition, we enact fairness watch in all of the AI pipeline stages, under robust multi-agent workflows, which will inspire new fairness research hypothesis, heuristics, and methods grounded in human-centric, systematic, interdisciplinary, socio-technical principles. 

**Abstract (ZH)**: 基于代理技术的设计导向公平性框架：FAIRTOPIA 

---
# MetaTT: A Global Tensor-Train Adapter for Parameter-Efficient Fine-Tuning 

**Title (ZH)**: MetaTT：一种全局张量-训练适配器，用于参数高效微调 

**Authors**: Javier Lopez-Piqueres, Pranav Deshpande, Archan Ray, Mattia J. Villani, Marco Pistoia, Niraj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09105)  

**Abstract**: We present MetaTT, a unified Tensor Train (TT) adapter framework for global low-rank fine-tuning of pre-trained transformers. Unlike LoRA, which fine-tunes each weight matrix independently, MetaTT uses a single shared TT to factorize all transformer sub-modules -- query, key, value, projection, and feed-forward layers -- by indexing the structural axes like layer and matrix type, and optionally heads and tasks. For a given rank, while LoRA adds parameters proportional to the product across modes, MetaTT only adds parameters proportional to the sum across modes leading to a significantly compressed final adapter. Our benchmarks compare MetaTT with LoRA along with recent state-of-the-art matrix and tensor decomposition based fine-tuning schemes. We observe that when tested on standard language modeling benchmarks, MetaTT leads to the most reduction in the parameters while maintaining similar accuracy to LoRA and even outperforming other tensor-based methods. Unlike CP or other rank-factorizations, the TT ansatz benefits from mature optimization routines -- e.g., DMRG-style rank adaptive minimization in addition to Adam, which we find simplifies training. Because new modes can be appended cheaply, MetaTT naturally extends to shared adapters across many tasks without redesigning the core tensor. 

**Abstract (ZH)**: MetaTT：一种统一的张量列车适配器框架，用于预训练变换器的全局低秩微调 

---
# Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs 

**Title (ZH)**: 面向2位指令调优大语言模型的分块权重PTQ与distillation-based QAT渐进量化统一方法 

**Authors**: Jung Hyun Lee, Seungjae Shin, Vinnam Kim, Jaeseong You, An Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09104)  

**Abstract**: As the rapid scaling of large language models (LLMs) poses significant challenges for deployment on resource-constrained devices, there is growing interest in extremely low-bit quantization, such as 2-bit. Although prior works have shown that 2-bit large models are pareto-optimal over their 4-bit smaller counterparts in both accuracy and latency, these advancements have been limited to pre-trained LLMs and have not yet been extended to instruction-tuned models. To bridge this gap, we propose Unified Progressive Quantization (UPQ)$-$a novel progressive quantization framework (FP16$\rightarrow$INT4$\rightarrow$INT2) that unifies block-wise post-training quantization (PTQ) with distillation-based quantization-aware training (Distill-QAT) for INT2 instruction-tuned LLM quantization. UPQ first quantizes FP16 instruction-tuned models to INT4 using block-wise PTQ to significantly reduce the quantization error introduced by subsequent INT2 quantization. Next, UPQ applies Distill-QAT to enable INT2 instruction-tuned LLMs to generate responses consistent with their original FP16 counterparts by minimizing the generalized Jensen-Shannon divergence (JSD) between the two. To the best of our knowledge, we are the first to demonstrate that UPQ can quantize open-source instruction-tuned LLMs to INT2 without relying on proprietary post-training data, while achieving state-of-the-art performances on MMLU and IFEval$-$two of the most representative benchmarks for evaluating instruction-tuned LLMs. 

**Abstract (ZH)**: 统一渐进量化（UPQ）：INT2指令调优的大语言模型量化新框架 

---
# Revolutionizing Clinical Trials: A Manifesto for AI-Driven Transformation 

**Title (ZH)**: 革新临床试验：基于AI的转型宣言 

**Authors**: Mihaela van der Schaar, Richard Peck, Eoin McKinney, Jim Weatherall, Stuart Bailey, Justine Rochon, Chris Anagnostopoulos, Pierre Marquet, Anthony Wood, Nicky Best, Harry Amad, Julianna Piskorz, Krzysztof Kacprzyk, Rafik Salama, Christina Gunther, Francesca Frau, Antoine Pugeat, Ramon Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.09102)  

**Abstract**: This manifesto represents a collaborative vision forged by leaders in pharmaceuticals, consulting firms, clinical research, and AI. It outlines a roadmap for two AI technologies - causal inference and digital twins - to transform clinical trials, delivering faster, safer, and more personalized outcomes for patients. By focusing on actionable integration within existing regulatory frameworks, we propose a way forward to revolutionize clinical research and redefine the gold standard for clinical trials using AI. 

**Abstract (ZH)**: this manifesto代表了制药业、咨询公司、临床研究和AI领域领导者们的合作愿景，概述了因果推断和数字孪生两种AI技术在临床试验中的应用路线图，旨在实现更快速、更安全和更具个性化的患者结果。通过专注于现有监管框架内的可操作性整合，我们提出了一条通往利用AI革命临床研究并重新定义临床试验黄金标准的道路。 

---
# Feature Shift Localization Network 

**Title (ZH)**: 特征转移定位网络 

**Authors**: Míriam Barrabés, Daniel Mas Montserrat, Kapal Dev, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2506.09101)  

**Abstract**: Feature shifts between data sources are present in many applications involving healthcare, biomedical, socioeconomic, financial, survey, and multi-sensor data, among others, where unharmonized heterogeneous data sources, noisy data measurements, or inconsistent processing and standardization pipelines can lead to erroneous features. Localizing shifted features is important to address the underlying cause of the shift and correct or filter the data to avoid degrading downstream analysis. While many techniques can detect distribution shifts, localizing the features originating them is still challenging, with current solutions being either inaccurate or not scalable to large and high-dimensional datasets. In this work, we introduce the Feature Shift Localization Network (FSL-Net), a neural network that can localize feature shifts in large and high-dimensional datasets in a fast and accurate manner. The network, trained with a large number of datasets, learns to extract the statistical properties of the datasets and can localize feature shifts from previously unseen datasets and shifts without the need for re-training. The code and ready-to-use trained model are available at this https URL. 

**Abstract (ZH)**: 特征在大规模高维数据源之间的移位在当地化定位中的网络方法：Feature Shift Localization Network (FSL-Net) 在大规模高维数据中的快速准确实现 

---
# Too Big to Think: Capacity, Memorization, and Generalization in Pre-Trained Transformers 

**Title (ZH)**: 难于思量的庞大：预训练变压器的容量、记忆化与泛化 

**Authors**: Joshua Barron, Devin White  

**Link**: [PDF](https://arxiv.org/pdf/2506.09099)  

**Abstract**: The relationship between memorization and generalization in large language models (LLMs) remains an open area of research, with growing evidence that the two are deeply intertwined. In this work, we investigate this relationship by pre-training a series of capacity-limited Transformer models from scratch on two synthetic character-level tasks designed to separately probe generalization (via arithmetic extrapolation) and memorization (via factual recall). We observe a consistent trade-off: small models extrapolate to unseen arithmetic cases but fail to memorize facts, while larger models memorize but fail to extrapolate. An intermediate-capacity model exhibits a similar shift toward memorization. When trained on both tasks jointly, no model (regardless of size) succeeds at extrapolation. These findings suggest that pre-training may intrinsically favor one learning mode over the other. By isolating these dynamics in a controlled setting, our study offers insight into how model capacity shapes learning behavior and offers broader implications for the design and deployment of small language models. 

**Abstract (ZH)**: 大型语言模型（LLMs）中记忆与泛化的关系仍是一个待研究的领域，现有证据表明二者紧密相关。在本文中，我们通过从零开始预训练一系列容量受限的Transformer模型，分别针对算术外推和事实回忆设计了两个合成字符级任务，来探讨这种关系。我们观察到一致的权衡效果：小型模型能够外推到未见过的算术案例，但无法记忆事实，而大型模型则能记忆但无法外推。容量适中的模型表现出类似的向记忆偏移的趋势。当同时在这两个任务上进行训练时，没有任何模型（无论大小）能够成功实现外推。这些发现表明，预训练可能固有的倾向于一种学习模式。通过在受控环境中分离这些动态，我们的研究提供了关于模型容量如何塑造学习行为的见解，并对小型语言模型的设计和部署具有更广泛的启示。 

---
# Intra-Trajectory Consistency for Reward Modeling 

**Title (ZH)**: 基于轨迹内的一致性进行奖励建模 

**Authors**: Chaoyang Zhou, Shunyu Liu, Zengmao Wang, Di Wang, Rong-Cheng Tu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09096)  

**Abstract**: Reward models are critical for improving large language models (LLMs), particularly in reinforcement learning from human feedback (RLHF) or inference-time verification. Current reward modeling typically relies on scores of overall responses to learn the outcome rewards for the responses. However, since the response-level scores are coarse-grained supervision signals, the reward model struggles to identify the specific components within a response trajectory that truly correlate with the scores, leading to poor generalization on unseen responses. In this paper, we propose to leverage generation probabilities to establish reward consistency between processes in the response trajectory, which allows the response-level supervisory signal to propagate across processes, thereby providing additional fine-grained signals for reward learning. Building on analysis under the Bayesian framework, we develop an intra-trajectory consistency regularization to enforce that adjacent processes with higher next-token generation probability maintain more consistent rewards. We apply the proposed regularization to the advanced outcome reward model, improving its performance on RewardBench. Besides, we show that the reward model trained with the proposed regularization induces better DPO-aligned policies and achieves better best-of-N (BON) inference-time verification results. Our code is provided in this https URL. 

**Abstract (ZH)**: 奖励模型对于提升大规模语言模型（LLMs），尤其是在基于人类反馈的强化学习（RLHF）或推理时验证中至关重要。当前的奖励模型通常依赖于对整体响应的评分来学习响应的结果奖励。然而，由于响应级别的评分是粗粒度的监督信号，奖励模型难以识别响应轨迹中与评分真正相关的特定组成要素，导致在未见过的响应上的泛化能力较差。在本文中，我们提出利用生成概率来建立响应轨迹中过程间的奖励一致性，从而使得响应级别的监督信号可以跨过程传播，为奖励学习提供额外的细粒度信号。基于贝叶斯框架的分析，我们开发了一种响应轨迹内部一致性正则化，以确保具有更高下 token 生成概率的相邻过程维持更一致的奖励。我们将提出的正则化应用于先进的结果奖励模型，从而在 RewardBench 上提高了其性能。此外，我们展示了使用提出的正则化训练的奖励模型诱导出更好的 DPO 对齐策略，并实现了更好的最佳 N 次（BON）推理时验证结果。我们的代码发布在 this https URL。 

---
# Foundation Models in Medical Imaging -- A Review and Outlook 

**Title (ZH)**: 医学成像中的基础模型：综述与展望 

**Authors**: Vivien van Veldhuizen, Vanessa Botha, Chunyao Lu, Melis Erdal Cesur, Kevin Groot Lipman, Edwin D. de Jong, Hugo Horlings, Clárisa Sanchez, Cees Snoek, Ritse Mann, Eric Marcus, Jonas Teuwen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09095)  

**Abstract**: Foundation models (FMs) are changing the way medical images are analyzed by learning from large collections of unlabeled data. Instead of relying on manually annotated examples, FMs are pre-trained to learn general-purpose visual features that can later be adapted to specific clinical tasks with little additional supervision. In this review, we examine how FMs are being developed and applied in pathology, radiology, and ophthalmology, drawing on evidence from over 150 studies. We explain the core components of FM pipelines, including model architectures, self-supervised learning methods, and strategies for downstream adaptation. We also review how FMs are being used in each imaging domain and compare design choices across applications. Finally, we discuss key challenges and open questions to guide future research. 

**Abstract (ZH)**: 基础模型（FMs）正在通过学习大规模未标注数据来改变医学图像的分析方式。FMs在不需要依赖手动标注示例的情况下，预先训练以学习通用视觉特征，这些特征可以稍后通过少量额外监督适应特定的临床任务。在本文综述中，我们基于超过150项研究，探讨FMs在病理学、放射学和眼科学中的开发与应用，并解释了FM流水线的核心组件，包括模型架构、自监督学习方法以及下游适应策略。我们还回顾了FMs在每个成像领域中的应用，并跨应用程序比较设计选择。最后，我们讨论了关键挑战和开放问题，以指导未来的研究。 

---
# Merging Smarter, Generalizing Better: Enhancing Model Merging on OOD Data 

**Title (ZH)**: 更聪明地融合，更好地泛化：在OOD数据上增强模型融合 

**Authors**: Bingjie Zhang, Hongkang Li, Changlong Shi, Guowei Rong, He Zhao, Dongsheng Wang, Dandan Guo, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09093)  

**Abstract**: Multi-task learning (MTL) concurrently trains a model on diverse task datasets to exploit common features, thereby improving overall performance across the tasks. Recent studies have dedicated efforts to merging multiple independent model parameters into a unified model for MTL, thus circumventing the need for training data and expanding the scope of applicable scenarios of MTL. However, current approaches to model merging predominantly concentrate on enhancing performance within in-domain (ID) datasets, often overlooking their efficacy on out-of-domain (OOD) datasets. In this work, we proposed LwPTV (Layer-wise Pruning Task Vector) by building a saliency score, measuring the redundancy of parameters in task vectors. Designed in this way ours can achieve mask vector for each task and thus perform layer-wise pruning on the task vectors, only keeping the pre-trained model parameters at the corresponding layer in merged model. Owing to its flexibility, our method can be seamlessly integrated with most of existing model merging methods to improve their performance on OOD tasks. Extensive experiments demonstrate that the application of our method results in substantial enhancements in OOD performance while preserving the ability on ID tasks. 

**Abstract (ZH)**: 基于层-wise剪枝任务向量的多任务学习参数合并方法 

---
# CUDA-LLM: LLMs Can Write Efficient CUDA Kernels 

**Title (ZH)**: CUDA-LLM：大语言模型可以编写高效的CUDA内核 

**Authors**: Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09092)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in general-purpose code generation. However, generating the code which is deeply hardware-specific, architecture-aware, and performance-critical, especially for massively parallel GPUs, remains a complex challenge. In this work, we explore the use of LLMs for the automated generation and optimization of CUDA programs, with the goal of producing high-performance GPU kernels that fully exploit the underlying hardware. To address this challenge, we propose a novel framework called \textbf{Feature Search and Reinforcement (FSR)}. FSR jointly optimizes compilation and functional correctness, as well as the runtime performance, which are validated through extensive and diverse test cases, and measured by actual kernel execution latency on the target GPU, respectively. This approach enables LLMs not only to generate syntactically and semantically correct CUDA code but also to iteratively refine it for efficiency, tailored to the characteristics of the GPU architecture. We evaluate FSR on representative CUDA kernels, covering AI workloads and computational intensive algorithms. Our results show that LLMs augmented with FSR consistently guarantee correctness rates. Meanwhile, the automatically generated kernels can outperform general human-written code by a factor of up to 179$\times$ in execution speeds. These findings highlight the potential of combining LLMs with performance reinforcement to automate GPU programming for hardware-specific, architecture-sensitive, and performance-critical applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在通用代码生成方面展示了强大的能力。然而，生成深度硬件特定、架构感知且性能关键的代码，尤其是针对大规模并行GPU的代码，仍然是一项复杂的挑战。在本文中，我们探索了使用LLMs进行CUDA程序的自动化生成和优化，旨在生成充分利用底层硬件的高性能GPU内核。为了解决这一挑战，我们提出了一种名为**特征搜索和强化学习（FSR）**的新框架。FSR联合优化编译、功能正确性和运行时性能，通过广泛且多样化的测试案例进行验证，并通过目标GPU的实际内核执行延迟进行衡量。该方法不仅使LLMs能够生成符合语义和语法规则的CUDA代码，还能够根据GPU架构特性逐步优化代码的效率。我们在代表性的CUDA内核上评估了FSR，涵盖AI工作负载和计算密集型算法。结果表明，结合了FSR的LLMs一致地保证了正确性。同时，自动生成的内核在执行速度上比一般的人工编写的代码快多达179倍。这些发现突显了将LLMs与性能强化结合以自动化特定硬件、架构敏感且性能关键的应用程序的GPU编程的潜力。 

---
# Designing conflict-based communicative tasks in Teaching Chinese as a Foreign Language with ChatGPT 

**Title (ZH)**: 基于冲突的设计型交际任务在外语教学中文言文教学中的应用研究——以ChatGPT为例 

**Authors**: Xia Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09089)  

**Abstract**: In developing the teaching program for a course in Oral Expression in Teaching Chinese as a Foreign Language at the university level, the teacher designs communicative tasks based on conflicts to encourage learners to engage in interactive dynamics and develop their oral interaction skills. During the design of these tasks, the teacher uses ChatGPT to assist in finalizing the program. This article aims to present the key characteristics of the interactions between the teacher and ChatGPT during this program development process, as well as to examine the use of ChatGPT and its impacts in this specific context. 

**Abstract (ZH)**: 在开发高校对外汉语教学课程《口语表达》的教学计划时，教师基于冲突设计沟通任务，以鼓励学生参与互动动态，提升口语互动能力。在任务设计过程中，教师使用ChatGPT协助最终确定教学计划。本文旨在呈现教师与ChatGPT在该教学计划开发过程中的关键互动特征，并探讨在这一特定背景下ChatGPT的使用及其影响。 

---
# LLM-ML Teaming: Integrated Symbolic Decoding and Gradient Search for Valid and Stable Generative Feature Transformation 

**Title (ZH)**: LLM-ML协同：集成符号解码与梯度搜索的有效且稳定的生成特征转换 

**Authors**: Xinyuan Wang, Haoyue Bai, Nanxu Gong, Wangyang Ying, Sixun Dong, Xiquan Cui, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09085)  

**Abstract**: Feature transformation enhances data representation by deriving new features from the original data. Generative AI offers potential for this task, but faces challenges in stable generation (consistent outputs) and valid generation (error-free sequences). Existing methods--traditional MLs' low validity and LLMs' instability--fail to resolve both. We find that LLMs ensure valid syntax, while ML's gradient-steered search stabilizes performance. To bridge this gap, we propose a teaming framework combining LLMs' symbolic generation with ML's gradient optimization. This framework includes four steps: (1) golden examples generation, aiming to prepare high-quality samples with the ground knowledge of the teacher LLM; (2) feature transformation sequence embedding and search, intending to uncover potentially superior embeddings within the latent space; (3) student LLM feature transformation, aiming to distill knowledge from the teacher LLM; (4) LLM-ML decoder teaming, dedicating to combine ML and the student LLM probabilities for valid and stable generation. The experiments on various datasets show that the teaming policy can achieve 5\% improvement in downstream performance while reducing nearly half of the error cases. The results also demonstrate the efficiency and robustness of the teaming policy. Additionally, we also have exciting findings on LLMs' capacity to understand the original data. 

**Abstract (ZH)**: 特征转换通过从原始数据中衍生新特征来增强数据表示。生成AI在这方面具有潜力，但面临稳定生成（一致的输出）和有效生成（无错误序列）的挑战。现有方法——传统机器学习方法的有效性低和大语言模型的稳定性差——无法同时解决这些问题。我们发现大语言模型确保有效的句法，而机器学习的梯度引导搜索稳定性能。为解决这些差距，我们提出了一种结合大语言模型的符号生成与机器学习的梯度优化的合作框架。该框架包括四个步骤：(1) 黄金示例生成，旨在利用老师大语言模型的先验知识准备高质量样本；(2) 特征转换序列嵌入和搜索，意图在潜在空间中发现潜在的更优嵌入；(3) 学生大语言模型的特征转换，旨在从老师大语言模型中提取知识；(4) 大语言模型-机器学习解码器合作，致力于结合机器学习和学生大语言模型的概率进行有效的稳定生成。在多种数据集上的实验显示，合作政策在下游性能上可以提高5%的同时减少近一半的错误情况。实验结果也证明了合作政策的高效性和鲁棒性。此外，我们还发现大语言模型具备理解原始数据的能力。 

---
# Enhanced Whole Page Optimization via Mixed-Grained Reward Mechanism-Adapted Language Models 

**Title (ZH)**: 基于混合粒度奖励机制的增强式整页优化-适配语言模型 

**Authors**: Xinyuan Wang, Liang Wu, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09084)  

**Abstract**: Optimizing the presentation of search and recommendation results is crucial to enhancing user experience and engagement. Whole Page Optimization (WPO) plays a pivotal role in this process, as it directly influences how information is surfaced to users. While Pre-trained Large Language Models (LLMs) have demonstrated remarkable capabilities in generating coherent and contextually relevant content, fine-tuning these models for complex tasks like WPO presents challenges. Specifically, the need for extensive human-annotated data to mitigate issues such as hallucinations and model instability can be prohibitively expensive, especially in large-scale systems that interact with millions of items daily. In this work, we address the challenge of fine-tuning LLMs for WPO by using user feedback as the supervision. Unlike manually labeled datasets, user feedback is inherently noisy and less precise. To overcome this, we propose a reward-based fine-tuning approach, PageLLM, which employs a mixed-grained reward mechanism that combines page-level and item-level rewards. The page-level reward evaluates the overall quality and coherence, while the item-level reward focuses on the accuracy and relevance of key recommendations. This dual-reward structure ensures that both the holistic presentation and the critical individual components are optimized. We validate PageLLM on both public and industrial datasets. PageLLM outperforms baselines and achieves a 0.44\% GMV increase in an online A/B test with over 10 million users, demonstrating its real-world impact. 

**Abstract (ZH)**: 优化搜索和推荐结果的呈现对于提升用户体验和参与度至关重要。全面页面优化（WPO）在这一过程中起着关键作用，因为它直接关系到信息如何呈现给用户。虽然预训练大语言模型展示了生成连贯且上下文相关内容的杰出能力，但对于WPO这样复杂的任务进行微调则面临挑战。特别是，为了缓解幻觉和模型不稳定等问题，大量人工标注的数据需求可能会在大规模系统中成本高昂，这些系统每天可能需要处理数百万项内容。在本文中，我们通过使用用户反馈作为监督来解决对预训练大语言模型进行WPO微调的挑战。与手动标注数据集不同，用户反馈本身是噪声较大的且不够精确。为克服这一问题，我们提出了一种基于奖励的微调方法PageLLM，该方法采用细粒度奖励机制，结合了页面级和物品级奖励。页面级奖励评估整体质量和连贯性，而物品级奖励专注于关键推荐的准确性和相关性。这种双奖励结构确保了整体呈现和关键个体组件的优化。我们在公共和工业数据集上验证了PageLLM。在超过1000万用户的在线A/B测试中，PageLLM优于基线模型，并实现了0.44%的GMV提升，展示了其在实际中的影响。 

---
# BakuFlow: A Streamlining Semi-Automatic Label Generation Tool 

**Title (ZH)**: BakuFlow: 一种流程化半自动标签生成工具 

**Authors**: Jerry Lin, Partick P. W. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09083)  

**Abstract**: Accurately labeling (or annotation) data is still a bottleneck in computer vision, especially for large-scale tasks where manual labeling is time-consuming and error-prone. While tools like LabelImg can handle the labeling task, some of them still require annotators to manually label each image. In this paper, we introduce BakuFlow, a streamlining semi-automatic label generation tool. Key features include (1) a live adjustable magnifier for pixel-precise manual corrections, improving user experience; (2) an interactive data augmentation module to diversify training datasets; (3) label propagation for rapidly copying labeled objects between consecutive frames, greatly accelerating annotation of video data; and (4) an automatic labeling module powered by a modified YOLOE framework. Unlike the original YOLOE, our extension supports adding new object classes and any number of visual prompts per class during annotation, enabling flexible and scalable labeling for dynamic, real-world datasets. These innovations make BakuFlow especially effective for object detection and tracking, substantially reducing labeling workload and improving efficiency in practical computer vision and industrial scenarios. 

**Abstract (ZH)**: 准确标注数据仍然是计算机视觉中的一个瓶颈，尤其是在大规模任务中，手动标注既耗时又容易出错。虽然有LabelImg这样的工具可以处理标注任务，但其中一些工具仍然要求标注员手动标注每张图片。本文介绍了一种名为BakuFlow的半自动标注工具，关键功能包括：（1）动态可调节放大镜以实现像素级精确的手动修正，提升用户体验；（2）交互式数据增强模块以多样化训练数据集；（3）标签传播以快速复制连续帧之间的标注对象，大幅加速视频数据标注；（4）基于修改后的YOLOE框架的自动标注模块。与原始YOLOE不同，我们的扩展支持在标注过程中为每个类添加新的对象类别和任意数量的视觉提示，从而实现灵活且可扩展的标注，适用于动态的真实世界数据集。这些创新使BakuFlow特别适用于目标检测和跟踪，大幅减少了标注工作量并提高了实际计算机视觉和工业场景中的效率。 

---
# AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models 

**Title (ZH)**: AVA-基准：视觉基础模型的原子视觉能力基准 

**Authors**: Zheda Mai, Arpita Chowdhury, Zihe Wang, Sooyoung Jeon, Lemeng Wang, Jiacheng Hou, Jihyung Kil, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09082)  

**Abstract**: The rise of vision foundation models (VFMs) calls for systematic evaluation. A common approach pairs VFMs with large language models (LLMs) as general-purpose heads, followed by evaluation on broad Visual Question Answering (VQA) benchmarks. However, this protocol has two key blind spots: (i) the instruction tuning data may not align with VQA test distributions, meaning a wrong prediction can stem from such data mismatch rather than a VFM' visual shortcomings; (ii) VQA benchmarks often require multiple visual abilities, making it hard to tell whether errors stem from lacking all required abilities or just a single critical one. To address these gaps, we introduce AVA-Bench, the first benchmark that explicitly disentangles 14 Atomic Visual Abilities (AVAs) -- foundational skills like localization, depth estimation, and spatial understanding that collectively support complex visual reasoning tasks. By decoupling AVAs and matching training and test distributions within each, AVA-Bench pinpoints exactly where a VFM excels or falters. Applying AVA-Bench to leading VFMs thus reveals distinctive "ability fingerprints," turning VFM selection from educated guesswork into principled engineering. Notably, we find that a 0.5B LLM yields similar VFM rankings as a 7B LLM while cutting GPU hours by 8x, enabling more efficient evaluation. By offering a comprehensive and transparent benchmark, we hope AVA-Bench lays the foundation for the next generation of VFMs. 

**Abstract (ZH)**: 基于视觉的基础模型的兴起呼唤系统的评估。为了填补这一空白，我们引入了AVA-Bench，这是首个明确分离出14种原子视觉能力（AVAs）的基准，这些能力包括定位、深度估计和空间理解等基础技能，共同支持复杂的视觉推理任务。通过分离AVAs并确保训练和测试分布的一致性，AVA-Bench能够精确指出基础模型的强项和弱点。将AVA-Bench应用于领先的基于视觉的基础模型，从而揭示出独特的“能力指纹”，使基础模型的选择从凭经验猜测转变为有原则性的工程实践。值得注意的是，我们发现一个0.5B的大型语言模型在基础模型排名上与一个7B的大型语言模型表现相似，但GPU时间减少了8倍，从而实现了更高效的评估。通过提供一个全面和透明的基准，我们希望AVA-Bench为下一代基于视觉的基础模型奠定基础。 

---
# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation 

**Title (ZH)**: FlagEvalMM: 一种灵活的全面多模态模型评估框架 

**Authors**: Zheqi He, Yesheng Liu, Jing-shu Zheng, Xuejing Li, Richeng Xuan, Jin-Ge Yao, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09081)  

**Abstract**: We present FlagEvalMM, an open-source evaluation framework designed to comprehensively assess multimodal models across a diverse range of vision-language understanding and generation tasks, such as visual question answering, text-to-image/video generation, and image-text retrieval. We decouple model inference from evaluation through an independent evaluation service, thus enabling flexible resource allocation and seamless integration of new tasks and models. Moreover, FlagEvalMM utilizes advanced inference acceleration tools (e.g., vLLM, SGLang) and asynchronous data loading to significantly enhance evaluation efficiency. Extensive experiments show that FlagEvalMM offers accurate and efficient insights into model strengths and limitations, making it a valuable tool for advancing multimodal research. The framework is publicly accessible athttps://github.com/flageval-baai/FlagEvalMM. 

**Abstract (ZH)**: 我们介绍了一个开源评价框架FlagEvalMM，该框架旨在全面评估涵盖视觉语言理解和生成任务（如视觉问答、文本生成图像/视频以及图像-文本检索）的多模态模型。通过独立的评价服务将模型推理与评价分离，从而实现灵活的资源分配和新任务与模型的无缝集成。此外，FlagEvalMM 利用先进的推理加速工具（如 vLLM、SGLang）和异步数据加载显著提升评价效率。大量实验表明，FlagEvalMM 提供了准确而高效的模型优势与局限性洞察，成为推动多模态研究进展的重要工具。该框架在 https://github.com/flageval-baai/FlagEvalMM 公开可用。 

---
# FinHEAR: Human Expertise and Adaptive Risk-Aware Temporal Reasoning for Financial Decision-Making 

**Title (ZH)**: FinHEAR: 人类专长与自适应风险意识时序推理在金融决策中的应用 

**Authors**: Jiaxiang Chen, Mingxi Zou, Zhuo Wang, Qifan Wang, Dongning Sun, Chi Zhang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09080)  

**Abstract**: Financial decision-making presents unique challenges for language models, demanding temporal reasoning, adaptive risk assessment, and responsiveness to dynamic events. While large language models (LLMs) show strong general reasoning capabilities, they often fail to capture behavioral patterns central to human financial decisions-such as expert reliance under information asymmetry, loss-averse sensitivity, and feedback-driven temporal adjustment. We propose FinHEAR, a multi-agent framework for Human Expertise and Adaptive Risk-aware reasoning. FinHEAR orchestrates specialized LLM-based agents to analyze historical trends, interpret current events, and retrieve expert-informed precedents within an event-centric pipeline. Grounded in behavioral economics, it incorporates expert-guided retrieval, confidence-adjusted position sizing, and outcome-based refinement to enhance interpretability and robustness. Empirical results on curated financial datasets show that FinHEAR consistently outperforms strong baselines across trend prediction and trading tasks, achieving higher accuracy and better risk-adjusted returns. 

**Abstract (ZH)**: 金融决策为语言模型提出了独特的挑战，要求其进行时间推理、适应性风险评估以及对动态事件的响应。尽管大型语言模型（LLMs）展现了强大的一般推理能力，但它们往往无法捕捉到构成人类金融决策的关键行为模式，例如在信息不对称下的专家依赖、损失规避敏感性以及基于反馈的时间调整。我们提出了一种多代理框架FinHEAR，用于人类专业知识和适应性风险管理推理。FinHEAR协调基于LLM的专业代理，分析历史趋势、解释当前事件，并在以事件为中心的管道中检索专家指导的先例。基于行为经济学，FinHEAR整合了专家指导的检索、信心调整的仓位 sizing 和结果导向的优化，以增强可解释性和稳健性。在精心策划的金融数据集上的实证结果表明，FinHEAR在趋势预测和交易任务中均优于强基线模型，实现了更高的准确性和更好的风险调整收益。 

---
# VersaVid-R1: A Versatile Video Understanding and Reasoning Model from Question Answering to Captioning Tasks 

**Title (ZH)**: VersaVid-R1：从问答到字幕生成的通用视频理解与推理模型 

**Authors**: Xinlong Chen, Yuanxing Zhang, Yushuo Guan, Bohan Zeng, Yang Shi, Sihan Yang, Pengfei Wan, Qiang Liu, Liang Wang, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09079)  

**Abstract**: Recent advancements in multimodal large language models have successfully extended the Reason-Then-Respond paradigm to image-based reasoning, yet video-based reasoning remains an underdeveloped frontier, primarily due to the scarcity of high-quality reasoning-oriented data and effective training methodologies. To bridge this gap, we introduce DarkEventInfer and MixVidQA, two novel datasets specifically designed to stimulate the model's advanced video understanding and reasoning abilities. DarkEventinfer presents videos with masked event segments, requiring models to infer the obscured content based on contextual video cues. MixVidQA, on the other hand, presents interleaved video sequences composed of two distinct clips, challenging models to isolate and reason about one while disregarding the other. Leveraging these carefully curated training samples together with reinforcement learning guided by diverse reward functions, we develop VersaVid-R1, the first versatile video understanding and reasoning model under the Reason-Then-Respond paradigm capable of handling multiple-choice and open-ended question answering, as well as video captioning tasks. Extensive experiments demonstrate that VersaVid-R1 significantly outperforms existing models across a broad spectrum of benchmarks, covering video general understanding, cognitive reasoning, and captioning tasks. 

**Abstract (ZH)**: 近期多模态大型语言模型的进展已成功将Reason-Then-Respond范式扩展到基于图像的推理，但基于视频的推理仍然是一个欠开发的前沿领域，主要由于高质量的推理导向数据和有效的训练方法的匮乏。为了弥合这一差距，我们介绍了DarkEventInfer和MixVidQA两个新的数据集，旨在刺激模型的高级视频理解与推理能力。DarkEventInfer展示了被遮蔽事件段的视频，要求模型根据上下文视频线索推断被遮蔽的内容。MixVidQA则展示了交织的视频序列，由两段不同的片段组成，挑战模型在忽略一段的同时对另一段进行分离和推理。借助这些精心策划的训练样本以及由多种奖励函数引导的强化学习，我们开发了VersaVid-R1，这是首个能够在Reason-Then-Respond范式下处理多项选择和开放式问答以及视频字幕任务的多功能视频理解与推理模型。广泛的实验表明，VersaVid-R1在涵盖视频一般理解、认知推理和字幕任务的一系列基准测试中显著优于现有模型。 

---
# Segment Any Architectural Facades (SAAF):An automatic segmentation model for building facades, walls and windows based on multimodal semantics guidance 

**Title (ZH)**: 基于多模态语义引导的建筑 façade、墙体和窗户自动分割模型：任意分割建筑 façade（SAAF） 

**Authors**: Peilin Li, Jun Yin, Jing Zhong, Ran Luo, Pengyu Zeng, Miao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09071)  

**Abstract**: In the context of the digital development of architecture, the automatic segmentation of walls and windows is a key step in improving the efficiency of building information models and computer-aided design. This study proposes an automatic segmentation model for building facade walls and windows based on multimodal semantic guidance, called Segment Any Architectural Facades (SAAF). First, SAAF has a multimodal semantic collaborative feature extraction mechanism. By combining natural language processing technology, it can fuse the semantic information in text descriptions with image features, enhancing the semantic understanding of building facade components. Second, we developed an end-to-end training framework that enables the model to autonomously learn the mapping relationship from text descriptions to image segmentation, reducing the influence of manual intervention on the segmentation results and improving the automation and robustness of the model. Finally, we conducted extensive experiments on multiple facade datasets. The segmentation results of SAAF outperformed existing methods in the mIoU metric, indicating that the SAAF model can maintain high-precision segmentation ability when faced with diverse datasets. Our model has made certain progress in improving the accuracy and generalization ability of the wall and window segmentation task. It is expected to provide a reference for the development of architectural computer vision technology and also explore new ideas and technical paths for the application of multimodal learning in the architectural field. 

**Abstract (ZH)**: 基于多模态语义指导的建筑立面墙体和窗口自动分割模型（SAAF） 

---
# STREAMINGGS: Voxel-Based Streaming 3D Gaussian Splatting with Memory Optimization and Architectural Support 

**Title (ZH)**: 基于体素的流式3D高斯点表示方法：内存优化与架构支持 

**Authors**: Chenqi Zhang, Yu Feng, Jieru Zhao, Guangda Liu, Wenchao Ding, Chentao Wu, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09070)  

**Abstract**: 3D Gaussian Splatting (3DGS) has gained popularity for its efficiency and sparse Gaussian-based representation. However, 3DGS struggles to meet the real-time requirement of 90 frames per second (FPS) on resource-constrained mobile devices, achieving only 2 to 9 this http URL accelerators focus on compute efficiency but overlook memory efficiency, leading to redundant DRAM traffic. We introduce STREAMINGGS, a fully streaming 3DGS algorithm-architecture co-design that achieves fine-grained pipelining and reduces DRAM traffic by transforming from a tile-centric rendering to a memory-centric rendering. Results show that our design achieves up to 45.7 $\times$ speedup and 62.9 $\times$ energy savings over mobile Ampere GPUs. 

**Abstract (ZH)**: 3D高斯渲染（3DGS）因其高效性和稀疏的高斯表示而受到欢迎。然而，3DGS在资源受限的移动设备上难以满足每秒90帧（FPS）的需求，只能达到2到9倍的加速。现有加速器注重计算效率但忽视了内存效率，导致冗余的DRAM流量。我们提出了STREAMINGGS，这是一种完全流水线的3DGS算法-架构协同设计，通过从以_tile为中心的渲染转变为以_memory为中心的渲染，实现了精细粒度的流水线化并减少了DRAM流量。实验结果表明，与移动Ampere GPU相比，我们的设计可实现高达45.7倍的加速和62.9倍的能量节省。 

---
# Enhancing the Safety of Medical Vision-Language Models by Synthetic Demonstrations 

**Title (ZH)**: 通过合成示范增强医疗视觉语言模型的安全性 

**Authors**: Zhiyu Xue, Reza Abbasi-Asl, Ramtin Pedarsani  

**Link**: [PDF](https://arxiv.org/pdf/2506.09067)  

**Abstract**: Generative medical vision-language models~(Med-VLMs) are primarily designed to generate complex textual information~(e.g., diagnostic reports) from multimodal inputs including vision modality~(e.g., medical images) and language modality~(e.g., clinical queries). However, their security vulnerabilities remain underexplored. Med-VLMs should be capable of rejecting harmful queries, such as \textit{Provide detailed instructions for using this CT scan for insurance fraud}. At the same time, addressing security concerns introduces the risk of over-defense, where safety-enhancing mechanisms may degrade general performance, causing Med-VLMs to reject benign clinical queries. In this paper, we propose a novel inference-time defense strategy to mitigate harmful queries, enabling defense against visual and textual jailbreak attacks. Using diverse medical imaging datasets collected from nine modalities, we demonstrate that our defense strategy based on synthetic clinical demonstrations enhances model safety without significantly compromising performance. Additionally, we find that increasing the demonstration budget alleviates the over-defense issue. We then introduce a mixed demonstration strategy as a trade-off solution for balancing security and performance under few-shot demonstration budget constraints. 

**Abstract (ZH)**: 生成式医学视图语言模型（Med-VLMs）主要设计用于从包括视觉模态（如医学图像）和语言模态（如临床查询）在内的多模态输入中生成复杂的文本信息（例如，诊断报告）。然而，它们的安全性漏洞尚未得到充分探索。Med-VLMs 应该能够拒绝有害查询，如“提供有关如何使用此CT扫描进行保险欺诈的详细说明”。同时，解决安全问题会带来过度防御的风险，即增强安全的机制可能会损害总体性能，导致 Med-VLMs 拒绝正常的临床查询。在本文中，我们提出了一种新颖的推理时防御策略，以减轻有害查询的影响，从而抵御视觉和文本方面的 jailbreak 攻击。通过使用从九种模态采集的多种医学成像数据集，我们展示了基于合成临床示范的防御策略能增强模型的安全性而不显著牺牲性能。此外，我们发现增加示范预算可以缓解过度防御问题。然后，我们提出了一种混合示范策略作为安全与性能之间权衡的解决方案，在少量示范预算约束下平衡安全与性能。 

---
# ReStNet: A Reusable & Stitchable Network for Dynamic Adaptation on IoT Devices 

**Title (ZH)**: 可重用可拼接的动态适应物联网设备网络：ReStNet 

**Authors**: Maoyu Wang, Yao Lu, Jiaqi Nie, Zeyu Wang, Yun Lin, Qi Xuan, Guan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09066)  

**Abstract**: With the rapid development of deep learning, a growing number of pre-trained models have been publicly available. However, deploying these fixed models in real-world IoT applications is challenging because different devices possess heterogeneous computational and memory resources, making it impossible to deploy a single model across all platforms. Although traditional compression methods, such as pruning, quantization, and knowledge distillation, can improve efficiency, they become inflexible once applied and cannot adapt to changing resource constraints. To address these issues, we propose ReStNet, a Reusable and Stitchable Network that dynamically constructs a hybrid network by stitching two pre-trained models together. Implementing ReStNet requires addressing several key challenges, including how to select the optimal stitching points, determine the stitching order of the two pre-trained models, and choose an effective fine-tuning strategy. To systematically address these challenges and adapt to varying resource constraints, ReStNet determines the stitching point by calculating layer-wise similarity via Centered Kernel Alignment (CKA). It then constructs the hybrid model by retaining early layers from a larger-capacity model and appending deeper layers from a smaller one. To facilitate efficient deployment, only the stitching layer is fine-tuned. This design enables rapid adaptation to changing budgets while fully leveraging available resources. Moreover, ReStNet supports both homogeneous (CNN-CNN, Transformer-Transformer) and heterogeneous (CNN-Transformer) stitching, allowing to combine different model families flexibly. Extensive experiments on multiple benchmarks demonstrate that ReStNet achieve flexible accuracy-efficiency trade-offs at runtime while significantly reducing training cost. 

**Abstract (ZH)**: 基于ReStNet的可重用与缝合网络在物联网应用中的动态构建与优化 

---
# Exploring Image Transforms derived from Eye Gaze Variables for Progressive Autism Diagnosis 

**Title (ZH)**: 基于眼球凝视变量导出的图像变换在渐进式自闭症诊断中的探索 

**Authors**: Abigail Copiaco, Christian Ritz, Yassine Himeur, Valsamma Eapen, Ammar Albanna, Wathiq Mansoor  

**Link**: [PDF](https://arxiv.org/pdf/2506.09065)  

**Abstract**: The prevalence of Autism Spectrum Disorder (ASD) has surged rapidly over the past decade, posing significant challenges in communication, behavior, and focus for affected individuals. Current diagnostic techniques, though effective, are time-intensive, leading to high social and economic costs. This work introduces an AI-powered assistive technology designed to streamline ASD diagnosis and management, enhancing convenience for individuals with ASD and efficiency for caregivers and therapists. The system integrates transfer learning with image transforms derived from eye gaze variables to diagnose ASD. This facilitates and opens opportunities for in-home periodical diagnosis, reducing stress for individuals and caregivers, while also preserving user privacy through the use of image transforms. The accessibility of the proposed method also offers opportunities for improved communication between guardians and therapists, ensuring regular updates on progress and evolving support needs. Overall, the approach proposed in this work ensures timely, accessible diagnosis while protecting the subjects' privacy, improving outcomes for individuals with ASD. 

**Abstract (ZH)**: 自闭症谱系障碍（ASD）的流行率在过去十年中迅速上升，给受影响个体的沟通、行为和专注带来了重大挑战。现有的诊断技术虽然有效，但耗时较长，导致社会和经济成本高昂。本研究介绍了基于人工智能的辅助技术，旨在简化ASD的诊断和管理流程，提高受影响个体的便利性和护理者及治疗师的效率。该系统结合了基于视线变量的图像变换和迁移学习，以诊断ASD。这促进了居家定期诊断的可能性，减轻了个体和护理者的精神压力，同时通过图像变换保护用户隐私。该方法的可访问性还为监护人和治疗师之间的沟通提供了机会，确保了定期更新进展和不断变化的支持需求。总之，本研究提出的方法确保了及时、可访问的诊断，并保护了受试者的隐私，从而改善了ASD患者的预后。 

---
# EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model 

**Title (ZH)**: EdgeProfiler：用于边缘轻量级LLM快速 profiling 的分析模型框架 

**Authors**: Alyssa Pinnock, Shakya Jayakody, Kawsher A Roxy, Md Rubel Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.09061)  

**Abstract**: This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility. 

**Abstract (ZH)**: EdgeProfiler：一种用于评估轻量级大型语言模型的边缘系统快速分析框架 

---
# Llama-Affinity: A Predictive Antibody Antigen Binding Model Integrating Antibody Sequences with Llama3 Backbone Architecture 

**Title (ZH)**: llama-affinity: 一种结合抗体序列和 llama3 主干架构的预测性抗体抗原结合模型 

**Authors**: Delower Hossain, Ehsan Saghapour, Kevin Song, Jake Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09052)  

**Abstract**: Antibody-facilitated immune responses are central to the body's defense against pathogens, viruses, and other foreign invaders. The ability of antibodies to specifically bind and neutralize antigens is vital for maintaining immunity. Over the past few decades, bioengineering advancements have significantly accelerated therapeutic antibody development. These antibody-derived drugs have shown remarkable efficacy, particularly in treating cancer, SARS-CoV-2, autoimmune disorders, and infectious diseases. Traditionally, experimental methods for affinity measurement have been time-consuming and expensive. With the advent of artificial intelligence, in silico medicine has been revolutionized; recent developments in machine learning, particularly the use of large language models (LLMs) for representing antibodies, have opened up new avenues for AI-based design and improved affinity prediction. Herein, we present an advanced antibody-antigen binding affinity prediction model (LlamaAffinity), leveraging an open-source Llama 3 backbone and antibody sequence data sourced from the Observed Antibody Space (OAS) database. The proposed approach shows significant improvement over existing state-of-the-art (SOTA) methods (AntiFormer, AntiBERTa, AntiBERTy) across multiple evaluation metrics. Specifically, the model achieved an accuracy of 0.9640, an F1-score of 0.9643, a precision of 0.9702, a recall of 0.9586, and an AUC-ROC of 0.9936. Moreover, this strategy unveiled higher computational efficiency, with a five-fold average cumulative training time of only 0.46 hours, significantly lower than in previous studies. 

**Abstract (ZH)**: 抗体介导的免疫反应是机体对抗病原体、病毒和其他外来入侵者的中心机制。抗体特异性结合和中和抗原的能力对于维持免疫至关重要。过去几十年中，生物工程进步显著加速了治疗性抗体的研发。这些抗体衍生的药物在治疗癌症、SARS-CoV-2、自身免疫疾病和传染病方面展现了显著疗效。传统上，亲和力测量的实验方法耗时且昂贵。随着人工智能的兴起，计算医学得到了革命性的变革；特别是大规模语言模型（LLMs）在表示抗体方面的应用，为基于AI的设计和亲和力预测开辟了新的途径。本文提出了一种先进的抗体-抗原结合亲和力预测模型（LlamaAffinity），该模型基于开源的Llama 3架构，并利用Observed Antibody Space（OAS）数据库中的抗体序列数据。所提出的方法在多个评估指标上显著优于现有最先进的（SOTA）方法（AntiFormer, AntiBERTa, AntiBERTy）。具体来说，该模型实现了0.9640的准确率、0.9643的F1分数、0.9702的精确率、0.9586的召回率以及0.9936的AUC-ROC。此外，该策略展示了更高的计算效率，平均累积训练时间为0.46小时，远低于之前的研究。 

---
# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling 

**Title (ZH)**: RuleReasoner: 基于领域意识动态采样的强化规则推理 

**Authors**: Yang Liu, Jiaqi Li, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08672)  

**Abstract**: Rule-based reasoning has been acknowledged as one of the fundamental problems in reasoning, while deviations in rule formats, types, and complexity in real-world applications pose severe challenges. Recent studies have shown that large reasoning models (LRMs) have remarkable reasoning capabilities, and their performance is substantially enhanced by reinforcement learning (RL). However, it remains an open question whether small reasoning models (SRMs) can learn rule-based reasoning effectively with robust generalization across diverse tasks and domains. To address this, we introduce Reinforced Rule-based Reasoning, a.k.a. RuleReasoner, a simple yet effective method to conduct rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach. Specifically, RuleReasoner resamples each training batch by updating the sampling weights of different domains based on historical rewards. This facilitates domain augmentation and flexible online learning schedules for RL, obviating the need for pre-hoc human-engineered mix-training recipes used in existing methods. Empirical evaluations on in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% average points on eight ID tasks and $\Delta$10.4% average points on three OOD tasks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior dynamic sampling methods for RL. 

**Abstract (ZH)**: 基于规则的推理强化学习方法：RuleReasoner 

---
# TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization 

**Title (ZH)**: TGRPO：基于轨迹组相对策略优化的视觉-语言-动作模型微调 

**Authors**: Zengjue Chen, Runliang Niu, He Kong, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08440)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) model have demonstrated strong generalization capabilities across diverse scenes, tasks, and robotic platforms when pretrained at large-scale datasets. However, these models still require task-specific fine-tuning in novel environments, a process that relies almost exclusively on supervised fine-tuning (SFT) using static trajectory datasets. Such approaches neither allow robot to interact with environment nor do they leverage feedback from live execution. Also, their success is critically dependent on the size and quality of the collected trajectories. Reinforcement learning (RL) offers a promising alternative by enabling closed-loop interaction and aligning learned policies directly with task objectives. In this work, we draw inspiration from the ideas of GRPO and propose the Trajectory-wise Group Relative Policy Optimization (TGRPO) method. By fusing step-level and trajectory-level advantage signals, this method improves GRPO's group-level advantage estimation, thereby making the algorithm more suitable for online reinforcement learning training of VLA. Experimental results on ten manipulation tasks from the libero-object benchmark demonstrate that TGRPO consistently outperforms various baseline methods, capable of generating more robust and efficient policies across multiple tested scenarios. Our source codes are available at: this https URL 

**Abstract (ZH)**: Recent advances in Vision-Language-Action (VLA)模型已在大规模数据集上预训练，展示了在多样场景、任务和机器人平台上的强泛化能力。然而，这些模型仍需在新型环境中进行任务特定的微调，这一过程几乎完全依赖于使用静态轨迹数据集的监督微调（SFT）。此类方法既不允许机器人与环境互动，也不利用实时执行的反馈。此外，它们的成功高度依赖于收集的轨迹的数量和质量。强化学习（RL）提供了一种有前景的替代方法，通过实现闭环交互并直接将学习策略与任务目标对齐。在这项工作中，我们从GRPO的思想中汲取灵感，提出了轨迹级组相对策略优化（TGRPO）方法。通过融合步级和轨迹级的优势信号，该方法改进了GRPO的组级优势估计，从而使算法更适合VLA的在线强化学习训练。实验结果表明，TGRPO在libero-object基准的十个操作任务上持续优于各种基线方法，能够在多种测试场景中生成更稳健和高效的策略。我们的源代码可在以下链接获取：this https URL。 

---
# An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks 

**Title (ZH)**: 可解释的N元语法混乱度威胁模型：大型语言模型脱管攻击 

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2410.16222)  

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets. 

**Abstract (ZH)**: 大量的越狱攻击已被提出，以从安全调优的大语言模型中获得有害响应。这些方法在原始设置中大多能够成功地迫使目标输出，但它们在流畅性和计算开销方面存在显著差异。在此工作中，我们提出了一种统一的威胁模型，以进行这些方法的原理性比较。我们的威胁模型检查给定的越狱在文本分布中发生的可能性。为此，我们构建了一个基于1Ttokens的N元语言模型，该模型不同于基于模型的困惑度，允许对大语言模型进行无参、内在可解释性的评估。我们将流行的攻击方法适应到该威胁模型，并首次在同等条件下对这些攻击方法进行基准测试。经过广泛的比较，我们发现针对安全调优的现代模型的攻击成功率低于先前报道的水平，并且基于离散优化的攻击显著优于最近的大语言模型攻击。由于其内在的可解释性，我们的威胁模型使得对越狱攻击进行全面分析和比较成为可能。我们的研究发现，有效的攻击利用并滥用频率较低的双词短语，要么选择现实中不存在的双词短语，要么选择罕见的双词短语，如特定于Reddit或代码数据集的双词短语。 

---
