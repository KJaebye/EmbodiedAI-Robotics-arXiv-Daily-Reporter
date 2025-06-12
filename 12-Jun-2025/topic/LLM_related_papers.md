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
# LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge 

**Title (ZH)**: LLMail-Inject：一项现实适应性提示注入挑战的数据集 

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09956)  

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection. 

**Abstract (ZH)**: 间接提示注入攻击利用了大规模语言模型（LLMs）区分输入中的指令与数据的固有限制。尽管提出了许多防御方案，针对适应性对手的系统评估仍然有限，即使成功的攻击可能具有广泛的安全和隐私影响，许多基于LLM的实际应用仍然易受攻击。我们介绍了LLMail-Inject公挑战的结果，该挑战模拟了一个现实场景，在该场景中，参与者适应性地尝试将恶意指令注入电子邮件，以触发基于LLM的电子邮件助手的未授权工具调用。该挑战涵盖了多种防御策略、LLM架构和检索配置，产生了包含839名参与者208,095个独特攻击提交的数据集。我们发布了挑战代码、完整的提交数据集以及我们的分析，展示了这些数据如何提供有关指令-数据区分问题的新见解。我们希望这将为未来针对提示注入的实际结构解决方案的研究奠定基础。 

---
# VerIF: Verification Engineering for Reinforcement Learning in Instruction Following 

**Title (ZH)**: VerIF: 强化学习在指令遵循中 struggl 工程验证 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09942)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become a key technique for enhancing large language models (LLMs), with verification engineering playing a central role. However, best practices for RL in instruction following remain underexplored. In this work, we explore the verification challenge in RL for instruction following and propose VerIF, a verification method that combines rule-based code verification with LLM-based verification from a large reasoning model (e.g., QwQ-32B). To support this approach, we construct a high-quality instruction-following dataset, VerInstruct, containing approximately 22,000 instances with associated verification signals. We apply RL training with VerIF to two models, achieving significant improvements across several representative instruction-following benchmarks. The trained models reach state-of-the-art performance among models of comparable size and generalize well to unseen constraints. We further observe that their general capabilities remain unaffected, suggesting that RL with VerIF can be integrated into existing RL recipes to enhance overall model performance. We have released our datasets, codes, and models to facilitate future research at this https URL. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已成为提升大规模语言模型（LLMs）的关键技术，验证工程起着核心作用。然而，指令跟随中的强化学习最佳实践仍待探索。本工作中，我们探讨了指令跟随中强化学习的验证挑战，并提出了一种结合规则基于的代码验证和大型推理模型（如QwQ-32B）基于的验证方法VerIF。为支持该方法，我们构建了一个高质量的指令跟随数据集VerInstruct，包含约22,000个实例及其关联的验证信号。我们将VerIF应用于两个模型，并在多个代表性指令跟随基准测试中取得了显著的改进。训练后的模型在可比较规模的模型中达到最佳性能，并且能够在未见过的约束下良好泛化。此外，我们观察到其通用能力未受影响，表明带有VerIF的强化学习可以集成到现有的强化学习配方中以提升整体模型性能。我们已公开了数据集、代码和模型以方便未来研究。 

---
# PersonaLens: A Benchmark for Personalization Evaluation in Conversational AI Assistants 

**Title (ZH)**: PersonaLens：面向对话AI助手个性化评价的标准基准 

**Authors**: Zheng Zhao, Clara Vania, Subhradeep Kayal, Naila Khan, Shay B. Cohen, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.09902)  

**Abstract**: Large language models (LLMs) have advanced conversational AI assistants. However, systematically evaluating how well these assistants apply personalization--adapting to individual user preferences while completing tasks--remains challenging. Existing personalization benchmarks focus on chit-chat, non-conversational tasks, or narrow domains, failing to capture the complexities of personalized task-oriented assistance. To address this, we introduce PersonaLens, a comprehensive benchmark for evaluating personalization in task-oriented AI assistants. Our benchmark features diverse user profiles equipped with rich preferences and interaction histories, along with two specialized LLM-based agents: a user agent that engages in realistic task-oriented dialogues with AI assistants, and a judge agent that employs the LLM-as-a-Judge paradigm to assess personalization, response quality, and task success. Through extensive experiments with current LLM assistants across diverse tasks, we reveal significant variability in their personalization capabilities, providing crucial insights for advancing conversational AI systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）已推动了对话式人工智能助手的发展。然而，系统地评估这些助手在个性化方面的能力——即根据不同用户偏好完成任务的能力——仍具有挑战性。现有的个性化基准主要集中在闲聊、非对话任务或狭窄领域，未能捕捉到个性化任务导向辅助的复杂性。为解决这一问题，我们引入了PersonaLens，一个全面的基准测试，用于评估任务导向人工智能助手的个性化能力。该基准包括多样化的用户画像，配备了丰富的偏好和交互历史，并配备了两个专门的LLM基代理：一个用户代理，与人工智能助手进行真实的任务导向对话；一个评判代理，采用LLM作为评判者的范式，评估个性化、响应质量和任务完成情况。通过广泛的实验，我们在多种任务中测评当前的LLM助手，揭示了他们在个性化能力方面的显著差异，提供了对推进对话式人工智能系统的重要见解。 

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
# Causal Sufficiency and Necessity Improves Chain-of-Thought Reasoning 

**Title (ZH)**: 因果充分性和必要性提高链式思维推理 

**Authors**: Xiangning Yu, Zhuohan Wang, Linyi Yang, Haoxuan Li, Anjie Liu, Xiao Xue, Jun Wang, Mengyue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09853)  

**Abstract**: Chain-of-Thought (CoT) prompting plays an indispensable role in endowing large language models (LLMs) with complex reasoning capabilities. However, CoT currently faces two fundamental challenges: (1) Sufficiency, which ensures that the generated intermediate inference steps comprehensively cover and substantiate the final conclusion; and (2) Necessity, which identifies the inference steps that are truly indispensable for the soundness of the resulting answer. We propose a causal framework that characterizes CoT reasoning through the dual lenses of sufficiency and necessity. Incorporating causal Probability of Sufficiency and Necessity allows us not only to determine which steps are logically sufficient or necessary to the prediction outcome, but also to quantify their actual influence on the final reasoning outcome under different intervention scenarios, thereby enabling the automated addition of missing steps and the pruning of redundant ones. Extensive experimental results on various mathematical and commonsense reasoning benchmarks confirm substantial improvements in reasoning efficiency and reduced token usage without sacrificing accuracy. Our work provides a promising direction for improving LLM reasoning performance and cost-effectiveness. 

**Abstract (ZH)**: 因果框架通过充足性和必要性的双重视角 karakterize CoT推理，结合因果概率的充足性和必要性，不仅可以确定哪些步骤对预测结果是逻辑上充分或必要的，还可以在不同干预场景下量化它们对最终推理结果的实际影响，从而实现缺失步骤的自动化添加和冗余步骤的修剪。在各种数学和常识推理基准测试上的广泛实验结果确认，在不牺牲准确性的情况下显著提高了推理效率并减少了标记使用量。我们的工作为提高LLM推理性能和成本效益提供了一个有希望的方向。 

---
# Superstudent intelligence in thermodynamics 

**Title (ZH)**: 超学生在热力学中的智能 

**Authors**: Rebecca Loubet, Pascal Zittlau, Marco Hoffmann, Luisa Vollmer, Sophie Fellenz, Heike Leitte, Fabian Jirasek, Johannes Lenhard, Hans Hasse  

**Link**: [PDF](https://arxiv.org/pdf/2506.09822)  

**Abstract**: In this short note, we report and analyze a striking event: OpenAI's large language model o3 has outwitted all students in a university exam on thermodynamics. The thermodynamics exam is a difficult hurdle for most students, where they must show that they have mastered the fundamentals of this important topic. Consequently, the failure rates are very high, A-grades are rare - and they are considered proof of the students' exceptional intellectual abilities. This is because pattern learning does not help in the exam. The problems can only be solved by knowledgeably and creatively combining principles of thermodynamics. We have given our latest thermodynamics exam not only to the students but also to OpenAI's most powerful reasoning model, o3, and have assessed the answers of o3 exactly the same way as those of the students. In zero-shot mode, the model o3 solved all problems correctly, better than all students who took the exam; its overall score was in the range of the best scores we have seen in more than 10,000 similar exams since 1985. This is a turning point: machines now excel in complex tasks, usually taken as proof of human intellectual capabilities. We discuss the consequences this has for the work of engineers and the education of future engineers. 

**Abstract (ZH)**: 开放AI的大语言模型o3在热力学大学考试中击败所有学生：一项引人注目的事件及其影响 

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
# TRIDENT: Temporally Restricted Inference via DFA-Enhanced Neural Traversal 

**Title (ZH)**: TRIDENT：基于DFA增强神经遍历的 temporally受限推理 

**Authors**: Vincenzo Collura, Karim Tit, Laura Bussi, Eleonora Giunchiglia, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.09701)  

**Abstract**: Large Language Models (LLMs) and other neural architectures have achieved impressive results across a variety of generative and classification tasks. However, they remain fundamentally ill-equipped to ensure that their outputs satisfy temporal constraints, such as those expressible in Linear Temporal Logic over finite traces (LTLf). In this paper, we introduce TRIDENT: a general and model-agnostic inference-time algorithm that guarantees compliance with such constraints without requiring any retraining. TRIDENT compiles LTLf formulas into a Deterministic Finite Automaton (DFA), which is used to guide a constrained variant of beam search. At each decoding step, transitions that would lead to constraint violations are masked, while remaining paths are dynamically re-ranked based on both the model's probabilities and the DFA's acceptance structure. We formally prove that the resulting sequences are guaranteed to satisfy the given LTLf constraints, and we empirically demonstrate that TRIDENT also improves output quality. We validate our approach on two distinct tasks: temporally constrained image-stream classification and controlled text generation. In both settings, TRIDENT achieves perfect constraint satisfaction, while comparison with the state of the art shows improved efficiency and high standard quality metrics. 

**Abstract (ZH)**: 大型语言模型(LLMs)和其他神经架构在生成和分类任务中取得了显著成果。然而，它们在确保其输出满足时间约束（如有限轨迹上的线性时态逻辑(LTLf)表达的时间约束）方面仍显得力不从心。本文介绍了TRIDENT：一个通用且模型无关的推理时算法，能够在无需重新训练的情况下保证输出满足此类约束。TRIDENT将LTLf公式编译为确定有限自动机(DFA)，用于指导受限变种的束搜索。在每次解码步骤中，会导致约束违反的过渡被掩藏，而剩余路径则基于模型概率和DFA接受结构动态重新排名。我们形式化证明了生成的序列保证满足给定的LTLf约束，并通过实验表明TRIDENT也提高了输出质量。我们分别在两个任务中验证了这种方法：时间受限的图像流分类和受控文本生成。在两种设置中，TRIDENT实现了完美的约束满足，与现有最佳方法相比，显示了更高的效率和高标准的质量指标。 

---
# Is Fine-Tuning an Effective Solution? Reassessing Knowledge Editing for Unstructured Data 

**Title (ZH)**: 微调是一个有效的解决方案吗？重新评估知识编辑对无结构数据的有效性 

**Authors**: Hao Xiong, Chuanyuan Tan, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09672)  

**Abstract**: Unstructured Knowledge Editing (UKE) is crucial for updating the relevant knowledge of large language models (LLMs). It focuses on unstructured inputs, such as long or free-form texts, which are common forms of real-world knowledge. Although previous studies have proposed effective methods and tested them, some issues exist: (1) Lack of Locality evaluation for UKE, and (2) Abnormal failure of fine-tuning (FT) based methods for UKE. To address these issues, we first construct two datasets, UnKEBench-Loc and AKEW-Loc (CF), by extending two existing UKE datasets with locality test data from the unstructured and structured views. This enables a systematic evaluation of the Locality of post-edited models. Furthermore, we identify four factors that may affect the performance of FT-based methods. Based on these factors, we conduct experiments to determine how the well-performing FT-based methods should be trained for the UKE task, providing a training recipe for future research. Our experimental results indicate that the FT-based method with the optimal setting (FT-UKE) is surprisingly strong, outperforming the existing state-of-the-art (SOTA). In batch editing scenarios, FT-UKE shows strong performance as well, with its advantage over SOTA methods increasing as the batch size grows, expanding the average metric lead from +6.78% to +10.80% 

**Abstract (ZH)**: 无结构知识编辑（UKE）是大型语言模型（LLMs）更新相关知识的关键。它专注于无结构输入，如长文本或自由格式文本，这些都是现实世界知识的常见形式。尽管先前的研究提出了一些有效的方法并进行了测试，但仍存在一些问题：(1) 缺乏无结构知识编辑的局部性评估，(2) 以微调（FT）为基础的方法在无结构知识编辑中出现异常失败。为了解决这些问题，我们首先通过扩展两个现有的UKE数据集并加入无结构和结构视角的局部性测试数据，构建了两个新的数据集UnKEBench-Loc和AKEW-Loc（CF），从而系统性地评估后编辑模型的局部性。此外，我们识别出了可能影响基于微调方法性能的四个因素，并基于这些因素进行了实验，以确定这些表现良好的基于微调的方法应如何为UKE任务进行训练，提供未来研究的训练方案。实验结果表明，具有最佳设置的基于微调的方法（FT-UKE）表现异常强大，优于现有最佳方法（SOTA）。在批量编辑场景中，FT-UKE同样表现出色，随着批次大小的增加，其相对于SOTA方法的优势增加，平均度量值领先优势从+6.78%扩大到+10.80%。 

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
# ReasonMed: A 370K Multi-Agent Generated Dataset for Advancing Medical Reasoning 

**Title (ZH)**: ReasonMed: 一个包含37万多个agents生成的数据集，用于推动医疗推理研究 

**Authors**: Yu Sun, Xingyu Qian, Weiwen Xu, Hao Zhang, Chenghao Xiao, Long Li, Yu Rong, Wenbing Huang, Qifeng Bai, Tingyang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09513)  

**Abstract**: Though reasoning-based large language models (LLMs) have excelled in mathematics and programming, their capabilities in knowledge-intensive medical question answering remain underexplored. To address this, we introduce ReasonMed, the largest medical reasoning dataset, comprising 370k high-quality examples distilled from 1.7 million initial reasoning paths generated by various LLMs. ReasonMed is constructed through a \textit{multi-agent verification and refinement process}, where we design an \textit{Error Refiner} to enhance the reasoning paths by identifying and correcting error-prone steps flagged by a verifier. Leveraging ReasonMed, we systematically investigate best practices for training medical reasoning models and find that combining detailed Chain-of-Thought (CoT) reasoning with concise answer summaries yields the most effective fine-tuning strategy. Based on this strategy, we train ReasonMed-7B, which sets a new benchmark for sub-10B models, outperforming the prior best by 4.17\% and even exceeding LLaMA3.1-70B on PubMedQA by 4.60\%. 

**Abstract (ZH)**: 尽管基于推理的大语言模型（LLMs）在数学和编程方面表现出色，但它们在知识密集型医疗问题解答方面的能力仍较少被探索。为了解决这一问题，我们介绍了ReasonMed，这是目前最大的医疗推理数据集，包含370,000个高质量的例子，这些例子是从各种LLM生成的170万条初始推理路径中提炼出来的。ReasonMed通过一个“多智能体验证和精炼过程”构建起来，在这个过程中，我们设计了一个“错误精炼器”，通过识别并修正验证器标记的错误步骤，来增强推理路径。利用ReasonMed，我们系统地研究了训练医疗推理模型的最佳实践，并发现结合详细的思维链（CoT）推理与简洁的答案总结是最有效的微调策略。基于这种策略，我们训练了ReasonMed-7B，它为小于10B的模型设定了一个新的基准，相对于之前的最好成绩提升了4.17%，甚至在PubMedQA上超过了LLaMA3.1-70B 4.60%。 

---
# UniToMBench: Integrating Perspective-Taking to Improve Theory of Mind in LLMs 

**Title (ZH)**: UniToMBench: 将换位思考整合以提升预训练语言模型的理论思维能力 

**Authors**: Prameshwar Thiyagarajan, Vaishnavi Parimi, Shamant Sai, Soumil Garg, Zhangir Meirbek, Nitin Yarlagadda, Kevin Zhu, Chris Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.09450)  

**Abstract**: Theory of Mind (ToM), the ability to understand the mental states of oneself and others, remains a challenging area for large language models (LLMs), which often fail to predict human mental states accurately. In this paper, we introduce UniToMBench, a unified benchmark that integrates the strengths of SimToM and TOMBENCH to systematically improve and assess ToM capabilities in LLMs by integrating multi-interaction task designs and evolving story scenarios. Supported by a custom dataset of over 1,000 hand-written scenarios, UniToMBench combines perspective-taking techniques with diverse evaluation metrics to better stimulate social cognition in LLMs. Through evaluation, we observe that while models like GPT-4o and GPT-4o Mini show consistently high accuracy in tasks involving emotional and belief-related scenarios, with results usually above 80%, there is significant variability in their performance across knowledge-based tasks. These results highlight both the strengths and limitations of current LLMs in ToM-related tasks, underscoring the value of UniToMBench as a comprehensive tool for future development. Our code is publicly available here: this https URL. 

**Abstract (ZH)**: 理论心智（ToM）：统一基准UniToMBench通过集成SimToM和TOMBENCH的优势，系统地提升和评估LLMs的ToM能力，融合多交互任务设计和演进的故事场景。 

---
# GigaChat Family: Efficient Russian Language Modeling Through Mixture of Experts Architecture 

**Title (ZH)**: gigachat 家族：通过专家混合架构高效构建俄语语言模型 

**Authors**: GigaChat team, Mamedov Valentin, Evgenii Kosarev, Gregory Leleytner, Ilya Shchuckin, Valeriy Berezovskiy, Daniil Smirnov, Dmitry Kozlov, Sergei Averkiev, Lukyanenko Ivan, Aleksandr Proshunin, Ainur Israfilova, Ivan Baskov, Artem Chervyakov, Emil Shakirov, Mikhail Kolesov, Daria Khomich, Darya Latortseva, Sergei Porkhun, Yury Fedorov, Oleg Kutuzov, Polina Kudriavtseva, Sofiia Soldatova, Kolodin Egor, Stanislav Pyatkin, Dzmitry Menshykh, Grafov Sergei, Eldar Damirov, Karlov Vladimir, Ruslan Gaitukiev, Arkadiy Shatenov, Alena Fenogenova, Nikita Savushkin, Fedor Minkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09440)  

**Abstract**: Generative large language models (LLMs) have become crucial for modern NLP research and applications across various languages. However, the development of foundational models specifically tailored to the Russian language has been limited, primarily due to the significant computational resources required. This paper introduces the GigaChat family of Russian LLMs, available in various sizes, including base models and instruction-tuned versions. We provide a detailed report on the model architecture, pre-training process, and experiments to guide design choices. In addition, we evaluate their performance on Russian and English benchmarks and compare GigaChat with multilingual analogs. The paper presents a system demonstration of the top-performing models accessible via an API, a Telegram bot, and a Web interface. Furthermore, we have released three open GigaChat models in open-source (this https URL), aiming to expand NLP research opportunities and support the development of industrial solutions for the Russian language. 

**Abstract (ZH)**: 生成型大型语言模型（LLMs）已成为现代多语言NLP研究和应用的关键。然而，针对俄罗斯语言的基础模型开发受到限制，主要原因是对计算资源的需求巨大。本文介绍了GigaChat家族系列俄语LLMs，包括不同规模的基础模型和指令调优版本，并提供了详细的模型架构、预训练过程及实验结果，以指导设计选择。此外，我们在俄语和英语基准上评估了这些模型的性能，并将其与多语言对照组进行了比较。本文还展示了顶级模型通过API、Telegram bot和Web界面的系统演示。我们还发布了三个开源的GigaChat模型（请参见以下链接），以扩大NLP研究机会，并支持俄语语言的工业解决方案开发。 

---
# Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting 

**Title (ZH)**: 改进的监督 fine-tuning 方法以减轻大型语言模型的灾难性遗忘 

**Authors**: Fei Ding, Baiqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09428)  

**Abstract**: Supervised Fine-Tuning (SFT), while enhancing large language models(LLMs)' instruction-following capabilities and domain-specific task adaptability, often diminishes their general capabilities. Moreover, due to the inaccessibility of original pre-training data, catastrophic forgetting tends to be exacerbated when third-party practitioners implement SFT on open-sourced models. To address this challenge, we propose a novel, more cost-effective SFT method which could effectively reduce the risk of catastrophic forgetting without access to original SFT data. Our approach begins by reconstructing the likely SFT instruction distribution of the base model, followed by a multi-model screening process to select optimal data, which is then mixed with new data for SFT. Experimental results demonstrate that our method preserves generalization capabilities in general domains while improving task-specific performance. 

**Abstract (ZH)**: 监督微调（SFT）虽然增强了大型语言模型（LLMs）的指令遵循能力和领域特定任务的适应性，但往往会损害其通用能力。此外，由于无法访问原始预训练数据，第三方实践者在开源模型上实施SFT时，灾难性遗忘的问题往往会加剧。为此，我们提出了一种新颖且成本更低的SFT方法，能够在不访问原始SFT数据的情况下有效降低灾难性遗忘的风险。该方法首先重构基模型可能的SFT指令分布，然后通过多模型筛选过程选择最优数据，将其与新数据混合以进行SFT。实验结果表明，该方法在保持通用域上的泛化能力的同时，提高了任务特定性能。 

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
# "Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions 

**Title (ZH)**: “这是真正的人类同伴支持者吗？”：在LLM支持的互动中同伴支持者与专家之间的 mis 对齐 

**Authors**: Kellie Yu Hui Sim, Roy Ka-Wei Lee, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09354)  

**Abstract**: Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client, context-sensitive LLM-generated suggestions, and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 5 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care. 

**Abstract (ZH)**: AI驱动的解决方案扩展心理社会支持的 acces: 基于实证经验的同伴支持的机遇与挑战 

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
# FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models 

**Title (ZH)**: FLoRIST: 特殊值阈值化方法实现高效准确的联邦微调大语言模型 

**Authors**: Hariharan Ramesh, Jyotikrishna Dass  

**Link**: [PDF](https://arxiv.org/pdf/2506.09199)  

**Abstract**: Integrating Low-Rank Adaptation (LoRA) into federated learning offers a promising solution for parameter-efficient fine-tuning of Large Language Models (LLMs) without sharing local data. However, several methods designed for federated LoRA present significant challenges in balancing communication efficiency, model accuracy, and computational cost, particularly among heterogeneous clients. These methods either rely on simplistic averaging of local adapters, which introduces aggregation noise, require transmitting large stacked local adapters, leading to poor communication efficiency, or necessitate reconstructing memory-dense global weight-update matrix and performing computationally expensive decomposition to design client-specific low-rank adapters. In this work, we propose FLoRIST, a federated fine-tuning framework that achieves mathematically accurate aggregation without incurring high communication or computational overhead. Instead of constructing the full global weight-update matrix at the server, FLoRIST employs an efficient decomposition pipeline by performing singular value decomposition on stacked local adapters separately. This approach operates within a compact intermediate space to represent the accumulated information from local LoRAs. We introduce tunable singular value thresholding for server-side optimal rank selection to construct a pair of global low-rank adapters shared by all clients. Extensive empirical evaluations across multiple datasets and LLMs demonstrate that FLoRIST consistently strikes the best balance between superior communication efficiency and competitive performance in both homogeneous and heterogeneous setups. 

**Abstract (ZH)**: 将低秩适应（LoRA）集成到联邦学习中，为大规模语言模型（LLMs）的参数高效微调提供了有望的解决方案，无需共享本地数据。然而，为联邦LoRA设计的几种方法在平衡通信效率、模型准确性和计算成本方面面临着显著挑战，尤其是在异构客户端之间。这些方法要么依赖于简单的本地适配器平均，引入了聚合噪声，要么需要传输大型堆叠的本地适配器，导致较差的通信效率，要么需要重建密集的记忆权重更新矩阵，并进行计算昂贵的分解来设计客户端特定的低秩适配器。在本文中，我们提出了一种联邦微调框架FLoRIST，该框架在不引入高通信或计算开销的情况下实现了数学上准确的聚合。FLoRIST 服务器不构建完整的全局权重更新矩阵，而是通过分别对堆叠的本地适配器进行奇异值分解来使用高效的分解管道。该方法在紧凑的中间空间中表示来自本地LoRA的累积信息。我们引入了可调的奇异值阈值优化服务器端的秩选择，构建由所有客户端共享的一对全局低秩适配器。在多个数据集和大语言模型上的广泛实证评估表明，FLoRIST 在同构和异构设置中始终能够实现通信效率和性能的最优平衡。 

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
# LLM-as-a-qualitative-judge: automating error analysis in natural language generation 

**Title (ZH)**: LLM作为定性裁判：自动化自然语言生成中的错误分析 

**Authors**: Nadezhda Chirkova, Tunde Oluwaseyi Ajayi, Seth Aycock, Zain Muhammad Mujahid, Vladana Perlić, Ekaterina Borisova, Markarit Vartampetian  

**Link**: [PDF](https://arxiv.org/pdf/2506.09147)  

**Abstract**: Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 将大型语言模型（LLMs） prompting 生成文本进行评估，即LLM-as-a-judge，已成为自然语言生成（NLG）的标准评估方法，但主要作为一种定量工具，即以数值分数为主要输出。本文提出LLM-as-a-qualitative-judge，这是一种基于LLM的评估方法，主要输出是对NLG系统输出中常见问题类型的结构化报告。该方法旨在为开发者提供有关如何改进给定NLG系统的有意义见解，包括两个主要步骤：实例问题开放分析和使用直观累加算法对发现的问题进行聚类。我们还引入了一种评估所提方法的策略，并提供了来自12个NLG数据集的约300个问题注解实例的标注。结果显示，LLM-as-a-qualitative-judge在2/3的情况下正确识别了实例特定的问题，并能够生成类似于人工注释者所编写的错误类型报告。我们的代码和数据可在以下网址公开访问：this https URL。 

---
# Unifying Block-wise PTQ and Distillation-based QAT for Progressive Quantization toward 2-bit Instruction-Tuned LLMs 

**Title (ZH)**: 面向2位指令调优大语言模型的分块权重PTQ与distillation-based QAT渐进量化统一方法 

**Authors**: Jung Hyun Lee, Seungjae Shin, Vinnam Kim, Jaeseong You, An Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09104)  

**Abstract**: As the rapid scaling of large language models (LLMs) poses significant challenges for deployment on resource-constrained devices, there is growing interest in extremely low-bit quantization, such as 2-bit. Although prior works have shown that 2-bit large models are pareto-optimal over their 4-bit smaller counterparts in both accuracy and latency, these advancements have been limited to pre-trained LLMs and have not yet been extended to instruction-tuned models. To bridge this gap, we propose Unified Progressive Quantization (UPQ)$-$a novel progressive quantization framework (FP16$\rightarrow$INT4$\rightarrow$INT2) that unifies block-wise post-training quantization (PTQ) with distillation-based quantization-aware training (Distill-QAT) for INT2 instruction-tuned LLM quantization. UPQ first quantizes FP16 instruction-tuned models to INT4 using block-wise PTQ to significantly reduce the quantization error introduced by subsequent INT2 quantization. Next, UPQ applies Distill-QAT to enable INT2 instruction-tuned LLMs to generate responses consistent with their original FP16 counterparts by minimizing the generalized Jensen-Shannon divergence (JSD) between the two. To the best of our knowledge, we are the first to demonstrate that UPQ can quantize open-source instruction-tuned LLMs to INT2 without relying on proprietary post-training data, while achieving state-of-the-art performances on MMLU and IFEval$-$two of the most representative benchmarks for evaluating instruction-tuned LLMs. 

**Abstract (ZH)**: 统一渐进量化（UPQ）：INT2指令调优的大语言模型量化新框架 

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
# CUDA-LLM: LLMs Can Write Efficient CUDA Kernels 

**Title (ZH)**: CUDA-LLM：大语言模型可以编写高效的CUDA内核 

**Authors**: Wentao Chen, Jiace Zhu, Qi Fan, Yehan Ma, An Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09092)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in general-purpose code generation. However, generating the code which is deeply hardware-specific, architecture-aware, and performance-critical, especially for massively parallel GPUs, remains a complex challenge. In this work, we explore the use of LLMs for the automated generation and optimization of CUDA programs, with the goal of producing high-performance GPU kernels that fully exploit the underlying hardware. To address this challenge, we propose a novel framework called \textbf{Feature Search and Reinforcement (FSR)}. FSR jointly optimizes compilation and functional correctness, as well as the runtime performance, which are validated through extensive and diverse test cases, and measured by actual kernel execution latency on the target GPU, respectively. This approach enables LLMs not only to generate syntactically and semantically correct CUDA code but also to iteratively refine it for efficiency, tailored to the characteristics of the GPU architecture. We evaluate FSR on representative CUDA kernels, covering AI workloads and computational intensive algorithms. Our results show that LLMs augmented with FSR consistently guarantee correctness rates. Meanwhile, the automatically generated kernels can outperform general human-written code by a factor of up to 179$\times$ in execution speeds. These findings highlight the potential of combining LLMs with performance reinforcement to automate GPU programming for hardware-specific, architecture-sensitive, and performance-critical applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在通用代码生成方面展示了强大的能力。然而，生成深度硬件特定、架构感知且性能关键的代码，尤其是针对大规模并行GPU的代码，仍然是一项复杂的挑战。在本文中，我们探索了使用LLMs进行CUDA程序的自动化生成和优化，旨在生成充分利用底层硬件的高性能GPU内核。为了解决这一挑战，我们提出了一种名为**特征搜索和强化学习（FSR）**的新框架。FSR联合优化编译、功能正确性和运行时性能，通过广泛且多样化的测试案例进行验证，并通过目标GPU的实际内核执行延迟进行衡量。该方法不仅使LLMs能够生成符合语义和语法规则的CUDA代码，还能够根据GPU架构特性逐步优化代码的效率。我们在代表性的CUDA内核上评估了FSR，涵盖AI工作负载和计算密集型算法。结果表明，结合了FSR的LLMs一致地保证了正确性。同时，自动生成的内核在执行速度上比一般的人工编写的代码快多达179倍。这些发现突显了将LLMs与性能强化结合以自动化特定硬件、架构敏感且性能关键的应用程序的GPU编程的潜力。 

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
# AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models 

**Title (ZH)**: AVA-基准：视觉基础模型的原子视觉能力基准 

**Authors**: Zheda Mai, Arpita Chowdhury, Zihe Wang, Sooyoung Jeon, Lemeng Wang, Jiacheng Hou, Jihyung Kil, Wei-Lun Chao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09082)  

**Abstract**: The rise of vision foundation models (VFMs) calls for systematic evaluation. A common approach pairs VFMs with large language models (LLMs) as general-purpose heads, followed by evaluation on broad Visual Question Answering (VQA) benchmarks. However, this protocol has two key blind spots: (i) the instruction tuning data may not align with VQA test distributions, meaning a wrong prediction can stem from such data mismatch rather than a VFM' visual shortcomings; (ii) VQA benchmarks often require multiple visual abilities, making it hard to tell whether errors stem from lacking all required abilities or just a single critical one. To address these gaps, we introduce AVA-Bench, the first benchmark that explicitly disentangles 14 Atomic Visual Abilities (AVAs) -- foundational skills like localization, depth estimation, and spatial understanding that collectively support complex visual reasoning tasks. By decoupling AVAs and matching training and test distributions within each, AVA-Bench pinpoints exactly where a VFM excels or falters. Applying AVA-Bench to leading VFMs thus reveals distinctive "ability fingerprints," turning VFM selection from educated guesswork into principled engineering. Notably, we find that a 0.5B LLM yields similar VFM rankings as a 7B LLM while cutting GPU hours by 8x, enabling more efficient evaluation. By offering a comprehensive and transparent benchmark, we hope AVA-Bench lays the foundation for the next generation of VFMs. 

**Abstract (ZH)**: 基于视觉的基础模型的兴起呼唤系统的评估。为了填补这一空白，我们引入了AVA-Bench，这是首个明确分离出14种原子视觉能力（AVAs）的基准，这些能力包括定位、深度估计和空间理解等基础技能，共同支持复杂的视觉推理任务。通过分离AVAs并确保训练和测试分布的一致性，AVA-Bench能够精确指出基础模型的强项和弱点。将AVA-Bench应用于领先的基于视觉的基础模型，从而揭示出独特的“能力指纹”，使基础模型的选择从凭经验猜测转变为有原则性的工程实践。值得注意的是，我们发现一个0.5B的大型语言模型在基础模型排名上与一个7B的大型语言模型表现相似，但GPU时间减少了8倍，从而实现了更高效的评估。通过提供一个全面和透明的基准，我们希望AVA-Bench为下一代基于视觉的基础模型奠定基础。 

---
# EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model 

**Title (ZH)**: EdgeProfiler：用于边缘轻量级LLM快速 profiling 的分析模型框架 

**Authors**: Alyssa Pinnock, Shakya Jayakody, Kawsher A Roxy, Md Rubel Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.09061)  

**Abstract**: This paper introduces EdgeProfiler, a fast profiling framework designed for evaluating lightweight Large Language Models (LLMs) on edge systems. While LLMs offer remarkable capabilities in natural language understanding and generation, their high computational, memory, and power requirements often confine them to cloud environments. EdgeProfiler addresses these challenges by providing a systematic methodology for assessing LLM performance in resource-constrained edge settings. The framework profiles compact LLMs, including TinyLLaMA, Gemma3.1B, Llama3.2-1B, and DeepSeek-r1-1.5B, using aggressive quantization techniques and strict memory constraints. Analytical modeling is used to estimate latency, FLOPs, and energy consumption. The profiling reveals that 4-bit quantization reduces model memory usage by approximately 60-70%, while maintaining accuracy within 2-5% of full-precision baselines. Inference speeds are observed to improve by 2-3x compared to FP16 baselines across various edge devices. Power modeling estimates a 35-50% reduction in energy consumption for INT4 configurations, enabling practical deployment on hardware such as Raspberry Pi 4/5 and Jetson Orin Nano Super. Our findings emphasize the importance of efficient profiling tailored to lightweight LLMs in edge environments, balancing accuracy, energy efficiency, and computational feasibility. 

**Abstract (ZH)**: EdgeProfiler：一种用于评估轻量级大型语言模型的边缘系统快速分析框架 

---
# An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks 

**Title (ZH)**: 可解释的N元语法混乱度威胁模型：大型语言模型脱管攻击 

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2410.16222)  

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets. 

**Abstract (ZH)**: 大量的越狱攻击已被提出，以从安全调优的大语言模型中获得有害响应。这些方法在原始设置中大多能够成功地迫使目标输出，但它们在流畅性和计算开销方面存在显著差异。在此工作中，我们提出了一种统一的威胁模型，以进行这些方法的原理性比较。我们的威胁模型检查给定的越狱在文本分布中发生的可能性。为此，我们构建了一个基于1Ttokens的N元语言模型，该模型不同于基于模型的困惑度，允许对大语言模型进行无参、内在可解释性的评估。我们将流行的攻击方法适应到该威胁模型，并首次在同等条件下对这些攻击方法进行基准测试。经过广泛的比较，我们发现针对安全调优的现代模型的攻击成功率低于先前报道的水平，并且基于离散优化的攻击显著优于最近的大语言模型攻击。由于其内在的可解释性，我们的威胁模型使得对越狱攻击进行全面分析和比较成为可能。我们的研究发现，有效的攻击利用并滥用频率较低的双词短语，要么选择现实中不存在的双词短语，要么选择罕见的双词短语，如特定于Reddit或代码数据集的双词短语。 

---
